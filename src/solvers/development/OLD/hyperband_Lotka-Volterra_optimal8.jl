using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux, ComponentArrays, Zygote
using Plots
using Statistics, Random
using StableRNGs
using LinearAlgebra
using Printf
using DataFrames
using LineSearches

# Set random seed for reproducibility
Random.seed!(1234)

# -----------------------------------------------------------------------------
# Setup the Lotka-Volterra System (using your exact configuration)
# -----------------------------------------------------------------------------

# Set a random seed for reproducible behaviour
rng = StableRNGs.StableRNG(1111)

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

# Extended timespan for extrapolation testing
tspan_extended = (0.0, 8.0)  # Extend beyond training data
t_extended = 0.0:0.25:8.0
prob_extended = ODEProblem(lotka!, u0, tspan_extended, p_)
solution_extended = solve(prob_extended, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = t_extended)
X_true_extended = Array(solution_extended)

# Plot the initial data
p_data = plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(p_data, t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])
display(p_data)

# -----------------------------------------------------------------------------
# BASELINE CONFIGURATION (Your provided setup)
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("TRAINING BASELINE MODEL (Manual Configuration)")
println("="^60)

rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const U_baseline = Lux.Chain(
    Lux.Dense(2, 5, rbf), 
    Lux.Dense(5, 5, rbf), 
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2)
)

# Get the initial parameters and state variables of the model
p_baseline, st_baseline = Lux.setup(rng, U_baseline)
const _st_baseline = st_baseline

# Define the hybrid model
function ude_dynamics_baseline!(du, u, p, t, p_true)
    û = U_baseline(u, p, _st_baseline)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics_baseline!(du, u, p, t) = ude_dynamics_baseline!(du, u, p, t, p_)
# Define the problem
prob_nn_baseline = ODEProblem(nn_dynamics_baseline!, Xₙ[:, 1], tspan, p_baseline)

function predict_baseline(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn_baseline, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

function loss_baseline(θ)
    X̂ = predict_baseline(θ)
    mean(abs2, Xₙ .- X̂)
end

losses_baseline = Float64[]
timing_baseline = Dict()

callback_baseline = function (state, l)
    push!(losses_baseline, l)
    if length(losses_baseline) % 50 == 0
        println("Current loss after $(length(losses_baseline)) iterations: $(losses_baseline[end])")
    end
    return false
end

# Training with ADAM + LBFGS
println("Stage 1: ADAM optimization...")
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_baseline(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_baseline))

# Capture timing and allocations
stats_baseline_adam = @timed res1 = solve(optprob, OptimizationOptimisers.Adam(), callback = callback_baseline, maxiters = 5000)
timing_baseline[:adam] = stats_baseline_adam.time
timing_baseline[:adam_bytes] = stats_baseline_adam.bytes
timing_baseline[:adam_gctime] = stats_baseline_adam.gctime

println("Training loss after $(length(losses_baseline)) iterations: $(losses_baseline[end])")
println(@sprintf("ADAM: %.2f seconds (%.2f G allocations: %.3f GiB, %.2f%% gc time)", 
    stats_baseline_adam.time,
    stats_baseline_adam.bytes / 1e9,
    stats_baseline_adam.bytes / (1024^3),
    100 * stats_baseline_adam.gctime / stats_baseline_adam.time))

println("\nStage 2: LBFGS refinement...")
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
stats_baseline_lbfgs = @timed res2 = solve(optprob2, OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()), 
                   callback = callback_baseline, maxiters = 1000)
timing_baseline[:lbfgs] = stats_baseline_lbfgs.time
timing_baseline[:lbfgs_bytes] = stats_baseline_lbfgs.bytes
timing_baseline[:lbfgs_gctime] = stats_baseline_lbfgs.gctime

println("Final training loss after $(length(losses_baseline)) iterations: $(losses_baseline[end])")
println(@sprintf("LBFGS: %.2f seconds (%.2f G allocations: %.3f GiB, %.2f%% gc time)", 
    stats_baseline_lbfgs.time,
    stats_baseline_lbfgs.bytes / 1e9,
    stats_baseline_lbfgs.bytes / (1024^3),
    100 * stats_baseline_lbfgs.gctime / stats_baseline_lbfgs.time))

p_trained_baseline = res2.u

# -----------------------------------------------------------------------------
# HYPERBAND OPTIMIZED CONFIGURATION
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("TRAINING HYPERBAND OPTIMIZED MODEL")
println("="^60)

# Hyperband best configuration
const U_hyperband = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

p_hyperband, st_hyperband = Lux.setup(StableRNGs.StableRNG(1111), U_hyperband)
const _st_hyperband = st_hyperband

function ude_dynamics_hyperband!(du, u, p, t, p_true)
    û = U_hyperband(u, p, _st_hyperband)[1]
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

nn_dynamics_hyperband!(du, u, p, t) = ude_dynamics_hyperband!(du, u, p, t, p_)
prob_nn_hyperband = ODEProblem(nn_dynamics_hyperband!, Xₙ[:, 1], tspan, p_hyperband)

function predict_hyperband(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn_hyperband, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, AutoTsit5(Rosenbrock23()), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

function loss_hyperband(θ)
    X̂ = predict_hyperband(θ)
    mean(abs2, Xₙ .- X̂)
end

losses_hyperband = Float64[]
timing_hyperband = Dict()

callback_hyperband = function (state, l)
    push!(losses_hyperband, l)
    if length(losses_hyperband) % 100 == 0
        println("  Iteration $(length(losses_hyperband)): Loss = $l")
    end
    return false
end

# Train with Hyperband's learning rate - ADAM first
optf_hb = Optimization.OptimizationFunction((x, p) -> loss_hyperband(x), adtype)
optprob_hb = Optimization.OptimizationProblem(optf_hb, ComponentVector{Float64}(p_hyperband))

println("Stage 1: ADAM optimization with lr=0.00390...")
stats_hb_adam = @timed res_hb1 = solve(optprob_hb, OptimizationOptimisers.Adam(0.00390), 
                                        callback = callback_hyperband, maxiters = 1000)
timing_hyperband[:adam] = stats_hb_adam.time
timing_hyperband[:adam_bytes] = stats_hb_adam.bytes
timing_hyperband[:adam_gctime] = stats_hb_adam.gctime

println("ADAM Training loss: $(losses_hyperband[end])")
println(@sprintf("ADAM: %.2f seconds (%.2f G allocations: %.3f GiB, %.2f%% gc time)", 
    stats_hb_adam.time,
    stats_hb_adam.bytes / 1e9,
    stats_hb_adam.bytes / (1024^3),
    100 * stats_hb_adam.gctime / stats_hb_adam.time))

# LBFGS refinement
println("\nStage 2: LBFGS refinement...")
optprob_hb2 = Optimization.OptimizationProblem(optf_hb, res_hb1.u)
stats_hb_lbfgs = @timed res_hb2 = solve(optprob_hb2, 
                                         OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()), 
                                         callback = callback_hyperband, maxiters = 1000)
timing_hyperband[:lbfgs] = stats_hb_lbfgs.time
timing_hyperband[:lbfgs_bytes] = stats_hb_lbfgs.bytes
timing_hyperband[:lbfgs_gctime] = stats_hb_lbfgs.gctime

p_trained_hyperband = res_hb2.u
println("Final loss: $(losses_hyperband[end])")
println(@sprintf("LBFGS: %.2f seconds (%.2f G allocations: %.3f GiB, %.2f%% gc time)", 
    stats_hb_lbfgs.time,
    stats_hb_lbfgs.bytes / 1e9,
    stats_hb_lbfgs.bytes / (1024^3),
    100 * stats_hb_lbfgs.gctime / stats_hb_lbfgs.time))

# -----------------------------------------------------------------------------
# RANDOM SEARCH OPTIMIZED CONFIGURATION
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("TRAINING RANDOM SEARCH OPTIMIZED MODEL")
println("="^60)

# Random Search best configuration
const U_random = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

p_random, st_random = Lux.setup(StableRNGs.StableRNG(1111), U_random)
const _st_random = st_random

function ude_dynamics_random!(du, u, p, t, p_true)
    û = U_random(u, p, _st_random)[1]
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

nn_dynamics_random!(du, u, p, t) = ude_dynamics_random!(du, u, p, t, p_)
prob_nn_random = ODEProblem(nn_dynamics_random!, Xₙ[:, 1], tspan, p_random)

function predict_random(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn_random, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

function loss_random(θ)
    X̂ = predict_random(θ)
    mean(abs2, Xₙ .- X̂)
end

losses_random = Float64[]
timing_random = Dict()

callback_random = function (state, l)
    push!(losses_random, l)
    if length(losses_random) % 100 == 0
        println("  Iteration $(length(losses_random)): Loss = $l")
    end
    return false
end

# Train with Random Search's learning rate - ADAM first
optf_rs = Optimization.OptimizationFunction((x, p) -> loss_random(x), adtype)
optprob_rs = Optimization.OptimizationProblem(optf_rs, ComponentVector{Float64}(p_random))

println("Stage 1: ADAM optimization with lr=0.000323...")
stats_rs_adam = @timed res_rs1 = solve(optprob_rs, OptimizationOptimisers.Adam(0.000323), 
                                        callback = callback_random, maxiters = 1000)
timing_random[:adam] = stats_rs_adam.time
timing_random[:adam_bytes] = stats_rs_adam.bytes
timing_random[:adam_gctime] = stats_rs_adam.gctime

println("ADAM Training loss: $(losses_random[end])")
println(@sprintf("ADAM: %.2f seconds (%.2f G allocations: %.3f GiB, %.2f%% gc time)", 
    stats_rs_adam.time,
    stats_rs_adam.bytes / 1e9,
    stats_rs_adam.bytes / (1024^3),
    100 * stats_rs_adam.gctime / stats_rs_adam.time))

# LBFGS refinement
println("\nStage 2: LBFGS refinement...")
optprob_rs2 = Optimization.OptimizationProblem(optf_rs, res_rs1.u)
stats_rs_lbfgs = @timed res_rs2 = solve(optprob_rs2, 
                                         OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()), 
                                         callback = callback_random, maxiters = 1000)
timing_random[:lbfgs] = stats_rs_lbfgs.time
timing_random[:lbfgs_bytes] = stats_rs_lbfgs.bytes
timing_random[:lbfgs_gctime] = stats_rs_lbfgs.gctime

p_trained_random = res_rs2.u
println("Final loss: $(losses_random[end])")
println(@sprintf("LBFGS: %.2f seconds (%.2f G allocations: %.3f GiB, %.2f%% gc time)", 
    stats_rs_lbfgs.time,
    stats_rs_lbfgs.bytes / 1e9,
    stats_rs_lbfgs.bytes / (1024^3),
    100 * stats_rs_lbfgs.gctime / stats_rs_lbfgs.time))

# Total training statistics
println("\n" * "="^60)
println("TOTAL TRAINING STATISTICS")
println("="^60)

total_time_baseline = timing_baseline[:adam] + timing_baseline[:lbfgs]
total_bytes_baseline = timing_baseline[:adam_bytes] + timing_baseline[:lbfgs_bytes]
total_gctime_baseline = timing_baseline[:adam_gctime] + timing_baseline[:lbfgs_gctime]

total_time_hyperband = timing_hyperband[:adam] + timing_hyperband[:lbfgs]
total_bytes_hyperband = timing_hyperband[:adam_bytes] + timing_hyperband[:lbfgs_bytes]
total_gctime_hyperband = timing_hyperband[:adam_gctime] + timing_hyperband[:lbfgs_gctime]

total_time_random = timing_random[:adam] + timing_random[:lbfgs]
total_bytes_random = timing_random[:adam_bytes] + timing_random[:lbfgs_bytes]
total_gctime_random = timing_random[:adam_gctime] + timing_random[:lbfgs_gctime]

println("BASELINE: $(round(total_time_baseline, digits=2)) seconds ($(round(total_bytes_baseline/1e9, digits=2)) G allocations: $(round(total_bytes_baseline/(1024^3), digits=3)) GiB, $(round(100*total_gctime_baseline/total_time_baseline, digits=2))% gc time)")
println("HYPERBAND: $(round(total_time_hyperband, digits=2)) seconds ($(round(total_bytes_hyperband/1e9, digits=2)) G allocations: $(round(total_bytes_hyperband/(1024^3), digits=3)) GiB, $(round(100*total_gctime_hyperband/total_time_hyperband, digits=2))% gc time)")
println("RANDOM SEARCH: $(round(total_time_random, digits=2)) seconds ($(round(total_bytes_random/1e9, digits=2)) G allocations: $(round(total_bytes_random/(1024^3), digits=3)) GiB, $(round(100*total_gctime_random/total_time_random, digits=2))% gc time)")

# -----------------------------------------------------------------------------
# COMPARISON ANALYSIS
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("COMPARATIVE ANALYSIS")
println("="^60)

# Generate predictions for comparison
ts = first(solution.t):(mean(diff(solution.t)) / 2):last(solution.t)
X̂_baseline = predict_baseline(p_trained_baseline, Xₙ[:, 1], ts)
X̂_hyperband = predict_hyperband(p_trained_hyperband, Xₙ[:, 1], ts)
X̂_random = predict_random(p_trained_random, Xₙ[:, 1], ts)

# Generate extended predictions for extrapolation testing
ts_extended = 0.0:0.125:8.0
try
    global X̂_baseline_ext = predict_baseline(p_trained_baseline, Xₙ[:, 1], ts_extended)
catch e
    println("Baseline extrapolation failed, using shorter range")
    global X̂_baseline_ext = predict_baseline(p_trained_baseline, Xₙ[:, 1], 0.0:0.125:6.0)
    global ts_extended = 0.0:0.125:6.0
end

try
    global X̂_hyperband_ext = predict_hyperband(p_trained_hyperband, Xₙ[:, 1], ts_extended)
catch e
    println("Hyperband extrapolation incomplete")
    global X̂_hyperband_ext = predict_hyperband(p_trained_hyperband, Xₙ[:, 1], 0.0:0.125:6.0)
end

try
    global X̂_random_ext = predict_random(p_trained_random, Xₙ[:, 1], ts_extended)
catch e
    println("Random extrapolation incomplete")
    global X̂_random_ext = predict_random(p_trained_random, Xₙ[:, 1], 0.0:0.125:6.0)
end

# Ensure all have same length
min_length = min(length(ts_extended), size(X̂_baseline_ext, 2), size(X̂_hyperband_ext, 2), size(X̂_random_ext, 2))
ts_extended = ts_extended[1:min_length]
X̂_baseline_ext = X̂_baseline_ext[:, 1:min_length]
X̂_hyperband_ext = X̂_hyperband_ext[:, 1:min_length]
X̂_random_ext = X̂_random_ext[:, 1:min_length]

# Calculate errors
train_idx = ts_extended .<= 5.0
test_idx = ts_extended .> 5.0

# Interpolate true solution for comparison
X_true_interp = zeros(2, length(ts_extended))
for (i, t) in enumerate(ts_extended)
    if t <= maximum(solution_extended.t)
        X_true_interp[:, i] = solution_extended(t)
    else
        X_true_interp[:, i] = solution_extended(maximum(solution_extended.t))
    end
end

# Training errors
train_error_baseline = mean(abs2, X_true_interp[:, train_idx] - X̂_baseline_ext[:, train_idx])
train_error_hyperband = mean(abs2, X_true_interp[:, train_idx] - X̂_hyperband_ext[:, train_idx])
train_error_random = mean(abs2, X_true_interp[:, train_idx] - X̂_random_ext[:, train_idx])

# Extrapolation errors (if available)
if any(test_idx)
    extrap_error_baseline = mean(abs2, X_true_interp[:, test_idx] - X̂_baseline_ext[:, test_idx])
    extrap_error_hyperband = mean(abs2, X_true_interp[:, test_idx] - X̂_hyperband_ext[:, test_idx])
    extrap_error_random = mean(abs2, X_true_interp[:, test_idx] - X̂_random_ext[:, test_idx])
else
    extrap_error_baseline = NaN
    extrap_error_hyperband = NaN
    extrap_error_random = NaN
end

println("\n📊 TRAINING ACCURACY (t ≤ 5.0):")
println("  Baseline MSE:      $(round(train_error_baseline, sigdigits=4))")
println("  Hyperband MSE:     $(round(train_error_hyperband, sigdigits=4))")
println("  Random Search MSE: $(round(train_error_random, sigdigits=4))")

if !isnan(extrap_error_baseline)
    println("\n🔮 EXTRAPOLATION ACCURACY (t > 5.0):")
    println("  Baseline MSE:      $(round(extrap_error_baseline, sigdigits=4))")
    println("  Hyperband MSE:     $(round(extrap_error_hyperband, sigdigits=4))")
    println("  Random Search MSE: $(round(extrap_error_random, sigdigits=4))")
end

# -----------------------------------------------------------------------------
# MISSING PHYSICS RECOVERY ANALYSIS
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("MISSING PHYSICS RECOVERY ANALYSIS")
println("="^60)

# Ideal unknown interactions
Ȳ_baseline = [-p_[2] * (X̂_baseline[1, :] .* X̂_baseline[2, :])'; 
               p_[3] * (X̂_baseline[1, :] .* X̂_baseline[2, :])']
Ȳ_hyperband = [-p_[2] * (X̂_hyperband[1, :] .* X̂_hyperband[2, :])'; 
                p_[3] * (X̂_hyperband[1, :] .* X̂_hyperband[2, :])']
Ȳ_random = [-p_[2] * (X̂_random[1, :] .* X̂_random[2, :])'; 
             p_[3] * (X̂_random[1, :] .* X̂_random[2, :])']

# Neural network predictions
Ŷ_baseline = U_baseline(X̂_baseline, p_trained_baseline, st_baseline)[1]
Ŷ_hyperband = U_hyperband(X̂_hyperband, p_trained_hyperband, st_hyperband)[1]
Ŷ_random = U_random(X̂_random, p_trained_random, st_random)[1]

# Calculate R² scores for missing physics recovery
function r2_score(y_true, y_pred)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    ss_res = sum((y_true - y_pred).^2)
    return 1 - ss_res/ss_tot
end

r2_baseline_1 = r2_score(vec(Ȳ_baseline[1, :]), vec(Ŷ_baseline[1, :]))
r2_baseline_2 = r2_score(vec(Ȳ_baseline[2, :]), vec(Ŷ_baseline[2, :]))
r2_hyperband_1 = r2_score(vec(Ȳ_hyperband[1, :]), vec(Ŷ_hyperband[1, :]))
r2_hyperband_2 = r2_score(vec(Ȳ_hyperband[2, :]), vec(Ŷ_hyperband[2, :]))
r2_random_1 = r2_score(vec(Ȳ_random[1, :]), vec(Ŷ_random[1, :]))
r2_random_2 = r2_score(vec(Ȳ_random[2, :]), vec(Ŷ_random[2, :]))

println("\n📈 R² SCORES FOR MISSING PHYSICS:")
println("  Baseline:      U₁ = $(round(r2_baseline_1, digits=4)), U₂ = $(round(r2_baseline_2, digits=4)), Avg = $(round((r2_baseline_1 + r2_baseline_2)/2, digits=4))")
println("  Hyperband:     U₁ = $(round(r2_hyperband_1, digits=4)), U₂ = $(round(r2_hyperband_2, digits=4)), Avg = $(round((r2_hyperband_1 + r2_hyperband_2)/2, digits=4))")
println("  Random Search: U₁ = $(round(r2_random_1, digits=4)), U₂ = $(round(r2_random_2, digits=4)), Avg = $(round((r2_random_1 + r2_random_2)/2, digits=4))")

# -----------------------------------------------------------------------------
# COMPREHENSIVE PERFORMANCE SUMMARY
# -----------------------------------------------------------------------------

# Calculate network parameters using parameterlength
n_params_baseline = Lux.parameterlength(U_baseline)
n_params_hyperband = Lux.parameterlength(U_hyperband)
n_params_random = Lux.parameterlength(U_random)

# Create hyperparameters summary - show all rows
hyperparams_summary = DataFrame(
    Configuration = ["Baseline (Manual)", "Hyperband", "Random Search"],
    Architecture = ["2→5→5→5→2", "2→32→32→32→32→32→2", "2→32→32→2"],
    Layers = [4, 6, 3],
    Parameters = [n_params_baseline, n_params_hyperband, n_params_random],
    Activation = ["RBF", "tanh", "tanh"],
    Learning_Rate = ["0.001 (default)", "0.00390", "0.000323"],
    Optimizer = ["ADAM→LBFGS", "ADAM→LBFGS", "ADAM→LBFGS"],
    ADAM_Iters = [5000, 1000, 1000],
    LBFGS_Iters = ["up to 1000", "up to 1000", "up to 1000"]
)

println("\n" * "="^60)
println("HYPERPARAMETERS SUMMARY")
println("="^60)
show(stdout, hyperparams_summary, allrows=true, allcols=true)
println()

# Performance metrics summary - show all rows
performance_summary = DataFrame(
    Metric = [
        "Final Training Loss",
        "Training MSE",
        "Extrapolation MSE",
        "Missing Physics R² (U₁)",
        "Missing Physics R² (U₂)",
        "Average R²",
        "Total Training Time (s)",
        "ADAM Time (s)",
        "LBFGS Time (s)",
        "Total Iterations",
        "Total Allocations (GiB)",
        "GC Time (%)"
    ],
    Baseline = [
        losses_baseline[end],
        train_error_baseline,
        isnan(extrap_error_baseline) ? NaN : extrap_error_baseline,
        r2_baseline_1,
        r2_baseline_2,
        (r2_baseline_1 + r2_baseline_2) / 2,
        round(total_time_baseline, digits=2),
        round(timing_baseline[:adam], digits=2),
        round(timing_baseline[:lbfgs], digits=2),
        length(losses_baseline),
        round(total_bytes_baseline/(1024^3), digits=3),
        round(100*total_gctime_baseline/total_time_baseline, digits=2)
    ],
    Hyperband = [
        losses_hyperband[end],
        train_error_hyperband,
        isnan(extrap_error_hyperband) ? NaN : extrap_error_hyperband,
        r2_hyperband_1,
        r2_hyperband_2,
        (r2_hyperband_1 + r2_hyperband_2) / 2,
        round(total_time_hyperband, digits=2),
        round(timing_hyperband[:adam], digits=2),
        round(timing_hyperband[:lbfgs], digits=2),
        length(losses_hyperband),
        round(total_bytes_hyperband/(1024^3), digits=3),
        round(100*total_gctime_hyperband/total_time_hyperband, digits=2)
    ],
    RandomSearch = [
        losses_random[end],
        train_error_random,
        isnan(extrap_error_random) ? NaN : extrap_error_random,
        r2_random_1,
        r2_random_2,
        (r2_random_1 + r2_random_2) / 2,
        round(total_time_random, digits=2),
        round(timing_random[:adam], digits=2),
        round(timing_random[:lbfgs], digits=2),
        length(losses_random),
        round(total_bytes_random/(1024^3), digits=3),
        round(100*total_gctime_random/total_time_random, digits=2)
    ]
)

println("\n" * "="^60)
println("PERFORMANCE METRICS SUMMARY")
println("="^60)
show(stdout, performance_summary, allrows=true, allcols=true)
println()

# Comparative improvements
println("\n" * "="^60)
println("COMPARATIVE IMPROVEMENTS")
println("="^60)

println("\n📊 Training Performance (vs Baseline):")
println("  Hyperband: $(round(train_error_baseline/train_error_hyperband, digits=2))x better")
println("  Random Search: $(round(train_error_baseline/train_error_random, digits=2))x better")
println("  Winner: $(train_error_hyperband < train_error_random ? "Hyperband" : "Random Search")")

if !isnan(extrap_error_baseline)
    println("\n🔮 Extrapolation Performance (vs Baseline):")
    println("  Hyperband: $(round(extrap_error_baseline/extrap_error_hyperband, digits=2))x better")
    println("  Random Search: $(round(extrap_error_baseline/extrap_error_random, digits=2))x better")
    println("  Winner: $(extrap_error_hyperband < extrap_error_random ? "Hyperband" : "Random Search")")
end

avg_r2_baseline = (r2_baseline_1 + r2_baseline_2) / 2
avg_r2_hyperband = (r2_hyperband_1 + r2_hyperband_2) / 2
avg_r2_random = (r2_random_1 + r2_random_2) / 2

println("\n🧠 Missing Physics Recovery (Average R²):")
println("  Baseline: $(round(avg_r2_baseline, digits=4))")
println("  Hyperband: $(round(avg_r2_hyperband, digits=4))")
println("  Random Search: $(round(avg_r2_random, digits=4))")
println("  Winner: $(argmax([avg_r2_baseline, avg_r2_hyperband, avg_r2_random]) == 1 ? "Baseline" : 
                      argmax([avg_r2_baseline, avg_r2_hyperband, avg_r2_random]) == 2 ? "Hyperband" : "Random Search")")

println("\n⏱️ Training Efficiency:")
println("  Baseline: $(round(total_time_baseline, digits=2))s for $(length(losses_baseline)) iterations")
println("  Hyperband: $(round(total_time_hyperband, digits=2))s for $(length(losses_hyperband)) iterations")
println("  Random Search: $(round(total_time_random, digits=2))s for $(length(losses_random)) iterations")
println("  Fastest: $(argmin([total_time_baseline, total_time_hyperband, total_time_random]) == 1 ? "Baseline" : 
                      argmin([total_time_baseline, total_time_hyperband, total_time_random]) == 2 ? "Hyperband" : "Random Search")")

println("\n💾 Model Complexity:")
println("  Baseline: $(n_params_baseline) parameters")
println("  Hyperband: $(n_params_hyperband) parameters ($(round(n_params_hyperband/n_params_baseline, digits=1))x baseline)")
println("  Random Search: $(n_params_random) parameters ($(round(n_params_random/n_params_baseline, digits=1))x baseline)")

# Final ranking
println("\n🏆 OVERALL RANKING:")
scores = Dict(
    "Baseline" => 0,
    "Hyperband" => 0,
    "Random Search" => 0
)

# Score based on final loss
if losses_baseline[end] < losses_hyperband[end] && losses_baseline[end] < losses_random[end]
    scores["Baseline"] += 1
elseif losses_hyperband[end] < losses_random[end]
    scores["Hyperband"] += 1
else
    scores["Random Search"] += 1
end

# Score based on training MSE
if train_error_baseline < train_error_hyperband && train_error_baseline < train_error_random
    scores["Baseline"] += 1
elseif train_error_hyperband < train_error_random
    scores["Hyperband"] += 1
else
    scores["Random Search"] += 1
end

# Score based on R²
if avg_r2_baseline > avg_r2_hyperband && avg_r2_baseline > avg_r2_random
    scores["Baseline"] += 1
elseif avg_r2_hyperband > avg_r2_random
    scores["Hyperband"] += 1
else
    scores["Random Search"] += 1
end

println("\nScores (out of 3 metrics):")
for (method, score) in scores
    println("  $method: $score")
end

winner = argmax(scores)
println("\n🥇 Winner: $winner")

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("CREATING VISUALIZATIONS")
println("="^60)

# Plot convergence comparison
pl_convergence = plot(title="Training Convergence Comparison", 
                      xlabel="Iterations", ylabel="Loss",
                      legend=:topright, size=(800, 600), yaxis=:log10)

# Plot all three methods
plot!(pl_convergence, 1:10:length(losses_baseline), losses_baseline[1:10:end], 
      label="Baseline (ADAM+LBFGS)", color=:green, linewidth=2)
plot!(pl_convergence, 1:10:length(losses_hyperband), losses_hyperband[1:10:end], 
      label="Hyperband (ADAM+LBFGS)", color=:blue, linewidth=2)
plot!(pl_convergence, 1:10:length(losses_random), losses_random[1:10:end], 
      label="Random Search (ADAM+LBFGS)", color=:red, linewidth=2)

# Mark transition points
vline!(pl_convergence, [5000], color=:green, linestyle=:dash, alpha=0.5, label="Baseline→LBFGS")
vline!(pl_convergence, [1000], color=:blue, linestyle=:dash, alpha=0.5, label="HB/RS→LBFGS")

display(pl_convergence)
savefig(pl_convergence, "convergence_comparison.png")

println("\n✅ Analysis complete! Results saved.")


# -----------------------------------------------------------------------------
# COMPREHENSIVE VISUALIZATION
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("CREATING COMPREHENSIVE VISUALIZATIONS")
println("="^60)

# Create a 2x3 subplot layout for comprehensive comparison
fig = plot(layout=(3,2), size=(1400, 1200), margin=5Plots.mm)

# 1. Population Dynamics Comparison
println("Creating population dynamics plot...")
plot!(fig[1], title="Population Dynamics Comparison", 
      xlabel="Time", ylabel="Population", legend=:topright)

# True solution
plot!(fig[1], solution_extended.t, X_true_extended[1, :], 
      label="True Prey", color=:black, linewidth=2, linestyle=:solid)
plot!(fig[1], solution_extended.t, X_true_extended[2, :], 
      label="True Predator", color=:gray, linewidth=2, linestyle=:solid)

# Predictions
plot!(fig[1], ts_extended, X̂_baseline_ext[1, :], 
      label="Baseline Prey", color=:green, linewidth=2, linestyle=:dash, alpha=0.7)
plot!(fig[1], ts_extended, X̂_baseline_ext[2, :], 
      label="Baseline Predator", color=:lightgreen, linewidth=2, linestyle=:dash, alpha=0.7)

plot!(fig[1], ts_extended, X̂_hyperband_ext[1, :], 
      label="Hyperband Prey", color=:blue, linewidth=2, linestyle=:dot, alpha=0.7)
plot!(fig[1], ts_extended, X̂_hyperband_ext[2, :], 
      label="Hyperband Predator", color=:lightblue, linewidth=2, linestyle=:dot, alpha=0.7)

plot!(fig[1], ts_extended, X̂_random_ext[1, :], 
      label="Random Prey", color=:red, linewidth=2, linestyle=:dashdot, alpha=0.7)
plot!(fig[1], ts_extended, X̂_random_ext[2, :], 
      label="Random Predator", color=:pink, linewidth=2, linestyle=:dashdot, alpha=0.7)

vline!(fig[1], [5.0], color=:orange, linestyle=:dash, linewidth=2, 
       label="Training|Extrapolation", alpha=0.5)

# 2. Phase Portrait Comparison
println("Creating phase portrait...")
plot!(fig[2], title="Phase Portrait Comparison", 
      xlabel="Prey", ylabel="Predator", legend=:topright)

plot!(fig[2], X_true_extended[1, :], X_true_extended[2, :], 
      label="Ground Truth", color=:black, linewidth=3)
plot!(fig[2], X̂_baseline_ext[1, :], X̂_baseline_ext[2, :], 
      label="Baseline", color=:green, linewidth=2, linestyle=:dash, alpha=0.8)
plot!(fig[2], X̂_hyperband_ext[1, :], X̂_hyperband_ext[2, :], 
      label="Hyperband", color=:blue, linewidth=2, linestyle=:dot, alpha=0.8)
plot!(fig[2], X̂_random_ext[1, :], X̂_random_ext[2, :], 
      label="Random Search", color=:red, linewidth=2, linestyle=:dashdot, alpha=0.8)

# 3. Training Convergence Comparison
println("Creating convergence plot...")
plot!(fig[3], title="Training Convergence Comparison", 
      xlabel="Iterations", ylabel="Loss", legend=:topright, yaxis=:log10)

plot!(fig[3], 1:10:length(losses_baseline), losses_baseline[1:10:end], 
      label="Baseline ADAM", color=:green, linewidth=2)
plot!(fig[3], 1:10:length(losses_hyperband), losses_hyperband[1:10:end], 
      label="Hyperband", color=:blue, linewidth=2)
plot!(fig[3], 1:10:length(losses_random), losses_random[1:10:end], 
      label="Random Search", color=:red, linewidth=2)

# 4. Prediction Error Over Time
println("Creating error plot...")
plot!(fig[4], title="Prediction Error Over Time", 
      xlabel="Time", ylabel="L2 Error", legend=:topleft, yaxis=:log)

error_baseline = vec(sqrt.(sum(abs2.(X_true_interp - X̂_baseline_ext), dims=1)))
error_hyperband = vec(sqrt.(sum(abs2.(X_true_interp - X̂_hyperband_ext), dims=1)))
error_random = vec(sqrt.(sum(abs2.(X_true_interp - X̂_random_ext), dims=1)))

plot!(fig[4], ts_extended, error_baseline, 
      label="Baseline", color=:green, linewidth=2)
plot!(fig[4], ts_extended, error_hyperband, 
      label="Hyperband", color=:blue, linewidth=2)
plot!(fig[4], ts_extended, error_random, 
      label="Random Search", color=:red, linewidth=2)
vline!(fig[4], [5.0], color=:orange, linestyle=:dash, linewidth=2, 
       label="Training|Extrapolation", alpha=0.5)

# 5. Missing Physics Recovery - Component 1
println("Creating missing physics recovery plots...")
plot!(fig[5], title="Missing Physics Recovery (U₁)", 
      xlabel="Time", ylabel="U₁(x,y)", legend=:topright)

plot!(fig[5], ts, vec(Ȳ_baseline[1, :]), 
      label="True U₁", color=:black, linewidth=3, alpha=0.5)
plot!(fig[5], ts, vec(Ŷ_baseline[1, :]), 
      label="Baseline U₁", color=:green, linewidth=2, linestyle=:dash)
plot!(fig[5], ts, vec(Ŷ_hyperband[1, :]), 
      label="Hyperband U₁", color=:blue, linewidth=2, linestyle=:dot)
plot!(fig[5], ts, vec(Ŷ_random[1, :]), 
      label="Random U₁", color=:red, linewidth=2, linestyle=:dashdot)

# 6. Missing Physics Recovery - Component 2
plot!(fig[6], title="Missing Physics Recovery (U₂)", 
      xlabel="Time", ylabel="U₂(x,y)", legend=:topright)

plot!(fig[6], ts, vec(Ȳ_baseline[2, :]), 
      label="True U₂", color=:black, linewidth=3, alpha=0.5)
plot!(fig[6], ts, vec(Ŷ_baseline[2, :]), 
      label="Baseline U₂", color=:lightgreen, linewidth=2, linestyle=:dash)
plot!(fig[6], ts, vec(Ŷ_hyperband[2, :]), 
      label="Hyperband U₂", color=:lightblue, linewidth=2, linestyle=:dot)
plot!(fig[6], ts, vec(Ŷ_random[2, :]), 
      label="Random U₂", color=:pink, linewidth=2, linestyle=:dashdot)

display(fig)
savefig(fig, "comprehensive_ude_comparison.png")

# Create separate detailed plots
println("\nCreating detailed individual plots...")

# Detailed Trajectory Comparison for Best Model (Hyperband)
p_best = plot(layout=(2,1), size=(1000, 800), margin=5Plots.mm)

plot!(p_best[1], title="Best Model (Hyperband) - Prey Dynamics", 
      xlabel="Time", ylabel="Prey Population", legend=:topright)
plot!(p_best[1], solution_extended.t, X_true_extended[1, :], 
      label="True", color=:black, linewidth=3)
plot!(p_best[1], ts_extended, X̂_hyperband_ext[1, :], 
      label="Hyperband Prediction", color=:blue, linewidth=2, linestyle=:dash)
scatter!(p_best[1], t, Xₙ[1, :], 
         label="Training Data", color=:red, markersize=3, alpha=0.6)
vline!(p_best[1], [5.0], color=:orange, linestyle=:dash, linewidth=2, 
       label="Training Boundary", alpha=0.5)

plot!(p_best[2], title="Best Model (Hyperband) - Predator Dynamics", 
      xlabel="Time", ylabel="Predator Population", legend=:topright)
plot!(p_best[2], solution_extended.t, X_true_extended[2, :], 
      label="True", color=:black, linewidth=3)
plot!(p_best[2], ts_extended, X̂_hyperband_ext[2, :], 
      label="Hyperband Prediction", color=:blue, linewidth=2, linestyle=:dash)
scatter!(p_best[2], t, Xₙ[2, :], 
         label="Training Data", color=:red, markersize=3, alpha=0.6)
vline!(p_best[2], [5.0], color=:orange, linestyle=:dash, linewidth=2, 
       label="Training Boundary", alpha=0.5)

display(p_best)
savefig(p_best, "best_model_detailed.png")

# Performance Metrics Bar Chart
println("\nCreating performance metrics bar chart...")
p_metrics = plot(layout=(2,2), size=(1200, 800), margin=5Plots.mm)

# Training Loss Comparison
methods = ["Baseline", "Hyperband", "Random"]
final_losses = [losses_baseline[end], losses_hyperband[end], losses_random[end]]
bar!(p_metrics[1], methods, final_losses, 
     title="Final Training Loss", ylabel="Loss", 
     color=[:green, :blue, :red], legend=false, yaxis=:log)

# MSE Comparison
mse_train = [train_error_baseline, train_error_hyperband, train_error_random]
mse_extrap = [extrap_error_baseline, extrap_error_hyperband, extrap_error_random]
groupedbar!(p_metrics[2], methods, [mse_train mse_extrap], 
            title="MSE Comparison", ylabel="MSE", 
            label=["Training" "Extrapolation"], 
            color=[:lightblue :darkblue], yaxis=:log)

# R² Scores
r2_scores = [(r2_baseline_1 + r2_baseline_2)/2, 
             (r2_hyperband_1 + r2_hyperband_2)/2, 
             (r2_random_1 + r2_random_2)/2]
bar!(p_metrics[3], methods, r2_scores, 
     title="Missing Physics R² (Average)", ylabel="R² Score", 
     color=[:green, :blue, :red], legend=false, ylims=(-5, 1))

# Training Time
train_times = [total_time_baseline, total_time_hyperband, total_time_random]
bar!(p_metrics[4], methods, train_times, 
     title="Total Training Time", ylabel="Time (seconds)", 
     color=[:green, :blue, :red], legend=false)

display(p_metrics)
savefig(p_metrics, "performance_metrics_comparison.png")

# Create a summary comparison table plot
println("\nCreating summary table visualization...")
summary_data = [
    "Final Loss" losses_baseline[end] losses_hyperband[end] losses_random[end];
    "Train MSE" train_error_baseline train_error_hyperband train_error_random;
    "Extrap MSE" extrap_error_baseline extrap_error_hyperband extrap_error_random;
    "Avg R²" (r2_baseline_1+r2_baseline_2)/2 (r2_hyperband_1+r2_hyperband_2)/2 (r2_random_1+r2_random_2)/2;
    "Parameters" n_params_baseline n_params_hyperband n_params_random;
    "Time (s)" total_time_baseline total_time_hyperband total_time_random
]

# Print the summary
println("\n" * "="^60)
println("FINAL SUMMARY TABLE")
println("="^60)
println("Metric          | Baseline    | Hyperband   | Random Search")
println("-"^60)
for i in 1:size(summary_data, 1)
    @printf("%-15s | %11.6f | %11.6f | %11.6f\n", 
            summary_data[i, 1], summary_data[i, 2], summary_data[i, 3], summary_data[i, 4])
end
println("="^60)

println("\n✅ All visualizations created and saved!")
println("Files saved:")
println("  - comprehensive_ude_comparison.png")
println("  - best_model_detailed.png")
println("  - performance_metrics_comparison.png")
println("  - convergence_comparison.png")