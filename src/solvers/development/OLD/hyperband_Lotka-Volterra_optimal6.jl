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

@time res1 = solve(optprob, OptimizationOptimisers.Adam(), callback = callback_baseline, maxiters = 5000)
println("Training loss after $(length(losses_baseline)) iterations: $(losses_baseline[end])")

println("Stage 2: LBFGS refinement...")
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
@time res2 = solve(optprob2, OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()), 
                   callback = callback_baseline, maxiters = 1000)
println("Final training loss after $(length(losses_baseline)) iterations: $(losses_baseline[end])")

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
callback_hyperband = function (state, l)
    push!(losses_hyperband, l)
    if length(losses_hyperband) % 100 == 0
        println("  Iteration $(length(losses_hyperband)): Loss = $l")
    end
    return false
end

# Train with Hyperband's learning rate
optf_hb = Optimization.OptimizationFunction((x, p) -> loss_hyperband(x), adtype)
optprob_hb = Optimization.OptimizationProblem(optf_hb, ComponentVector{Float64}(p_hyperband))

println("Training with learning rate: 0.00390")
@time res_hb = solve(optprob_hb, OptimizationOptimisers.Adam(0.00390), callback = callback_hyperband, maxiters = 1500)
p_trained_hyperband = res_hb.u
println("Final loss: $(losses_hyperband[end])")

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
callback_random = function (state, l)
    push!(losses_random, l)
    if length(losses_random) % 100 == 0
        println("  Iteration $(length(losses_random)): Loss = $l")
    end
    return false
end

# Train with Random Search's learning rate
optf_rs = Optimization.OptimizationFunction((x, p) -> loss_random(x), adtype)
optprob_rs = Optimization.OptimizationProblem(optf_rs, ComponentVector{Float64}(p_random))

println("Training with learning rate: 0.000323")
@time res_rs = solve(optprob_rs, OptimizationOptimisers.Adam(0.000323), callback = callback_random, maxiters = 1500)
p_trained_random = res_rs.u
println("Final loss: $(losses_random[end])")

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

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("CREATING VISUALIZATIONS")
println("="^60)

# Plot 1: Trajectory comparison
pl_trajectory = plot(title="Population Dynamics Comparison", 
                     xlabel="Time", ylabel="Population",
                     legend=:topright, size=(800, 500))

# True solution
plot!(pl_trajectory, solution_extended.t, X_true_extended[1, :], 
      label="True Prey", color=:black, linewidth=2, linestyle=:solid)
plot!(pl_trajectory, solution_extended.t, X_true_extended[2, :], 
      label="True Predator", color=:gray, linewidth=2, linestyle=:solid)

# Baseline predictions
plot!(pl_trajectory, ts_extended, X̂_baseline_ext[1, :], 
      label="Baseline Prey", color=:green, linewidth=2, linestyle=:dash)
plot!(pl_trajectory, ts_extended, X̂_baseline_ext[2, :], 
      label="Baseline Predator", color=:lightgreen, linewidth=2, linestyle=:dash)

# Hyperband predictions
plot!(pl_trajectory, ts_extended, X̂_hyperband_ext[1, :], 
      label="Hyperband Prey", color=:blue, linewidth=2, linestyle=:dot)
plot!(pl_trajectory, ts_extended, X̂_hyperband_ext[2, :], 
      label="Hyperband Predator", color=:lightblue, linewidth=2, linestyle=:dot)

# Random Search predictions
plot!(pl_trajectory, ts_extended, X̂_random_ext[1, :], 
      label="Random Prey", color=:red, linewidth=2, linestyle=:dashdot)
plot!(pl_trajectory, ts_extended, X̂_random_ext[2, :], 
      label="Random Predator", color=:pink, linewidth=2, linestyle=:dashdot)

# Mark training boundary
vline!(pl_trajectory, [5.0], color=:orange, linestyle=:dash, linewidth=2, 
       label="Training|Extrapolation")

# Plot 2: Phase portrait comparison
pl_phase = plot(title="Phase Portrait Comparison", 
                xlabel="Prey", ylabel="Predator",
                legend=:topright, size=(600, 600))

plot!(pl_phase, X_true_extended[1, :], X_true_extended[2, :], 
      label="Ground Truth", color=:black, linewidth=2)
plot!(pl_phase, X̂_baseline_ext[1, :], X̂_baseline_ext[2, :], 
      label="Baseline", color=:green, linewidth=2, linestyle=:dash)
plot!(pl_phase, X̂_hyperband_ext[1, :], X̂_hyperband_ext[2, :], 
      label="Hyperband", color=:blue, linewidth=2, linestyle=:dot)
plot!(pl_phase, X̂_random_ext[1, :], X̂_random_ext[2, :], 
      label="Random Search", color=:red, linewidth=2, linestyle=:dashdot)

# Plot 3: Training convergence comparison
pl_losses = plot(title="Training Convergence Comparison", 
                 xlabel="Iterations", ylabel="Loss",
                 legend=:topright, size=(600, 400), yaxis=:log10)

# Baseline: ADAM + LBFGS
plot!(pl_losses, 1:5000, losses_baseline[1:5000], 
      label="Baseline ADAM", color=:green, linewidth=2)
if length(losses_baseline) > 5000
    plot!(pl_losses, 5001:length(losses_baseline), losses_baseline[5001:end], 
          label="Baseline LBFGS", color=:darkgreen, linewidth=2)
end

# Hyperband
plot!(pl_losses, 1:10:length(losses_hyperband), losses_hyperband[1:10:end], 
      label="Hyperband", color=:blue, linewidth=2)

# Random Search
plot!(pl_losses, 1:10:length(losses_random), losses_random[1:10:end], 
      label="Random Search", color=:red, linewidth=2)

# Plot 4: Error over time
pl_error = plot(title="Prediction Error Over Time", 
                xlabel="Time", ylabel="L2 Error",
                legend=:topleft, size=(600, 400), yaxis=:log)

error_baseline = vec(sqrt.(sum(abs2.(X_true_interp - X̂_baseline_ext), dims=1)))
error_hyperband = vec(sqrt.(sum(abs2.(X_true_interp - X̂_hyperband_ext), dims=1)))
error_random = vec(sqrt.(sum(abs2.(X_true_interp - X̂_random_ext), dims=1)))

plot!(pl_error, ts_extended, error_baseline, 
      label="Baseline", color=:green, linewidth=2)
plot!(pl_error, ts_extended, error_hyperband, 
      label="Hyperband", color=:blue, linewidth=2)
plot!(pl_error, ts_extended, error_random, 
      label="Random Search", color=:red, linewidth=2)
vline!(pl_error, [5.0], color=:orange, linestyle=:dash, linewidth=2, 
       label="Training|Extrapolation")

# Combine plots
combined_plot = plot(pl_trajectory, pl_phase, pl_losses, pl_error, 
                     layout=(2,2), size=(1400, 1000))
display(combined_plot)
savefig(combined_plot, "baseline_vs_optimized_comparison.png")

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

# Plot missing physics recovery
pl_reconstruction = plot(ts, transpose(Ŷ_baseline), 
                        xlabel="Time", ylabel="U(x,y)", 
                        color=[:green :lightgreen], linewidth=2,
                        label=["Baseline U₁" "Baseline U₂"],
                        title="Missing Physics Recovery", linestyle=:solid)
plot!(pl_reconstruction, ts, transpose(Ŷ_hyperband), 
      color=[:blue :lightblue], linewidth=2,
      label=["Hyperband U₁" "Hyperband U₂"], linestyle=:dash)
plot!(pl_reconstruction, ts, transpose(Ŷ_random), 
      color=[:red :pink], linewidth=2,
      label=["Random U₁" "Random U₂"], linestyle=:dot)
plot!(pl_reconstruction, ts, transpose(Ȳ_baseline), 
      color=[:black :gray], linewidth=3, alpha=0.5,
      label=["True U₁" "True U₂"], linestyle=:solid)

display(pl_reconstruction)
savefig(pl_reconstruction, "missing_physics_comparison.png")

# -----------------------------------------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------------------------------------

summary = DataFrame(
    Metric = [
        "Final Training Loss",
        "Training MSE",
        "Extrapolation MSE",
        "Missing Physics R² (U₁)",
        "Missing Physics R² (U₂)",
        "Network Parameters",
        "Network Depth",
        "Activation Function",
        "Learning Rate",
        "Training Method"
    ],
    Baseline = [
        losses_baseline[end],
        train_error_baseline,
        isnan(extrap_error_baseline) ? "N/A" : extrap_error_baseline,
        r2_baseline_1,
        r2_baseline_2,
        sum(length, Lux.params(U_baseline)),
        4,  # 3 hidden + 1 output layer
        "RBF",
        "ADAM(0.001) → LBFGS",
        "Manual"
    ],
    Hyperband = [
        losses_hyperband[end],
        train_error_hyperband,
        isnan(extrap_error_hyperband) ? "N/A" : extrap_error_hyperband,
        r2_hyperband_1,
        r2_hyperband_2,
        sum(length, Lux.params(U_hyperband)),
        6,  # 5 hidden + 1 output layer
        "tanh",
        "0.00390",
        "Optimized"
    ],
    RandomSearch = [
        losses_random[end],
        train_error_random,
        isnan(extrap_error_random) ? "N/A" : extrap_error_random,
        r2_random_1,
        r2_random_2,
        sum(length, Lux.params(U_random)),
        3,  # 2 hidden + 1 output layer
        "tanh",
        "0.000323",
        "Optimized"
    ]
)

println("\n" * "="^60)
println("FINAL COMPARISON SUMMARY")
println("="^60)
display(summary)

# Calculate improvements
println("\n" * "="^60)
println("PERFORMANCE IMPROVEMENTS")
println("="^60)

println("\n📊 Training Performance:")
println("  Baseline → Hyperband: $(round(train_error_baseline/train_error_hyperband, digits=2))x improvement")
println("  Baseline → Random Search: $(round(train_error_baseline/train_error_random, digits=2))x improvement")
println("  Hyperband vs Random Search: $(round(train_error_random/train_error_hyperband, digits=2))x")

if !isnan(extrap_error_baseline)
    println("\n🔮 Extrapolation Performance:")
    println("  Baseline → Hyperband: $(round(extrap_error_baseline/extrap_error_hyperband, digits=2))x improvement")
    println("  Baseline → Random Search: $(round(extrap_error_baseline/extrap_error_random, digits=2))x improvement")
    println("  Hyperband vs Random Search: $(round(extrap_error_random/extrap_error_hyperband, digits=2))x")
end

println("\n🧠 Missing Physics Recovery (Average R²):")
avg_r2_baseline = (r2_baseline_1 + r2_baseline_2) / 2
avg_r2_hyperband = (r2_hyperband_1 + r2_hyperband_2) / 2
avg_r2_random = (r2_random_1 + r2_random_2) / 2
println("  Baseline: $(round(avg_r2_baseline, digits=4))")
println("  Hyperband: $(round(avg_r2_hyperband, digits=4))")
println("  Random Search: $(round(avg_r2_random, digits=4))")

println("\n✅ Comparison complete! Results saved to:")
println("  - baseline_vs_optimized_comparison.png")
println("  - missing_physics_comparison.png")