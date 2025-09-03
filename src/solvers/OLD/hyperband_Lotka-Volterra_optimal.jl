using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers
using Lux, ComponentArrays, Zygote
using Plots
using Statistics, Random
using Printf

# Set random seed for reproducibility
Random.seed!(1234)

# -----------------------------------------------------------------------------
# Setup the Lotka-Volterra System
# -----------------------------------------------------------------------------

# True system parameters
const p_true = Float32[1.3, 0.9, 0.8, 1.8]  # α, β, γ, δ

function lotka_true!(du, u, p, t)
    α, β, γ, δ = p
    x, y = u
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
end

# Initial condition
const u0 = Float32[0.44249296, 4.6280594]

# Training data timespan
const tspan_train = (0.0f0, 3.0f0)
const t_train = range(tspan_train[1], tspan_train[2], length=40)

# Extended timespan for prediction/extrapolation
const tspan_pred = (0.0f0, 6.0f0)  # Double the training time
const t_pred = range(tspan_pred[1], tspan_pred[2], length=80)

# Generate ground truth data
prob_true = ODEProblem(lotka_true!, u0, tspan_pred, p_true)
sol_true = solve(prob_true, Vern7(), abstol=1e-12, reltol=1e-12, saveat=t_pred)
X_true_full = Array(sol_true)

# Training data (with noise)
X_true_train = X_true_full[:, 1:40]
X_noisy = X_true_train + Float32(1e-3) * randn(Float32, size(X_true_train))

# -----------------------------------------------------------------------------
# Create and Train UDE Models with Different Hyperparameters
# -----------------------------------------------------------------------------

"""
Create a trained UDE model with specified hyperparameters
"""
function create_trained_ude(config, training_iterations=1000; verbose=false)
    # Extract hyperparameters
    hidden_dim = config[:hidden_dim]
    n_layers = config[:n_layers]
    activation = config[:activation]
    lr = config[:learning_rate]
    solver = config[:solver]
    
    # Build neural network
    layers = []
    push!(layers, Dense(3, hidden_dim, activation))  # Input: [x, y, t]
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim, hidden_dim, activation))
    end
    push!(layers, Dense(hidden_dim, 1))  # Output: missing term
    
    nn_model = Lux.Chain(layers...)
    
    rng = Random.default_rng()
    Random.seed!(rng, 1111)
    p_init, st = Lux.setup(rng, nn_model)
    
    # Define UDE dynamics
    function ude_dynamics!(du, u, p, t)
        û = nn_model([u[1], u[2], t], p, st)[1]
        du[1] = p_true[1] * u[1] - p_true[2] * u[1] * u[2]
        du[2] = -p_true[4] * u[2] + û[1]  # Neural network replaces γ*x*y term
    end
    
    # Create ODE problem
    prob_nn = ODEProblem(ude_dynamics!, u0, tspan_train, p_init)
    
    # Prediction function
    function predict(θ)
        _prob = remake(prob_nn, p=θ)
        Array(solve(_prob, solver, saveat=t_train,
                    abstol=1e-6, reltol=1e-6,
                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end
    
    # Loss function
    function loss(θ)
        X̂ = predict(θ)
        mean(abs2, X_noisy - X̂)
    end
    
    # Training
    losses = Float32[]
    callback = function (p, l)
        push!(losses, l)
        if verbose && length(losses) % 100 == 0
            println("  Iteration $(length(losses)): Loss = $l")
        end
        return false
    end
    
    # Optimize
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
    
    res = Optimization.solve(optprob, ADAM(lr), callback=callback, maxiters=training_iterations)
    
    # Return trained model and parameters
    return nn_model, res.u, st, losses, solver
end

"""
Make predictions with trained UDE model
"""
function predict_ude(nn_model, p_trained, st, solver, tspan, t_save)
    function ude_dynamics!(du, u, p, t)
        û = nn_model([u[1], u[2], t], p, st)[1]
        du[1] = p_true[1] * u[1] - p_true[2] * u[1] * u[2]
        du[2] = -p_true[4] * u[2] + û[1]
    end
    
    prob = ODEProblem(ude_dynamics!, u0, tspan, p_trained)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-6, reltol=1e-6)
    return Array(sol)
end

# -----------------------------------------------------------------------------
# Define Hyperparameters from Previous Results
# -----------------------------------------------------------------------------

# Hyperband best configuration
hyperband_config = Dict(
    :hidden_dim => 32,
    :n_layers => 5,
    :activation => tanh,
    :learning_rate => 0.00390,
    :solver => AutoTsit5(Rosenbrock23())
)

# Random Search best configuration
random_config = Dict(
    :hidden_dim => 32,
    :n_layers => 2,
    :activation => tanh,
    :learning_rate => 0.000323,
    :solver => Vern7()
)

# -----------------------------------------------------------------------------
# Train Both Models
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("TRAINING UDE MODELS")
println("="^60)

# Train Hyperband model
println("\n📊 Training UDE with Hyperband hyperparameters...")
println("  Architecture: $(hyperband_config[:n_layers]) layers × $(hyperband_config[:hidden_dim]) neurons")
println("  Learning rate: $(hyperband_config[:learning_rate])")
@time nn_hb, p_hb, st_hb, losses_hb, solver_hb = create_trained_ude(
    hyperband_config, 1500; verbose=true
)
println("  Final training loss: $(losses_hb[end])")

# Train Random Search model
println("\n🎲 Training UDE with Random Search hyperparameters...")
println("  Architecture: $(random_config[:n_layers]) layers × $(random_config[:hidden_dim]) neurons")
println("  Learning rate: $(random_config[:learning_rate])")
@time nn_rs, p_rs, st_rs, losses_rs, solver_rs = create_trained_ude(
    random_config, 1500; verbose=true
)
println("  Final training loss: $(losses_rs[end])")

# -----------------------------------------------------------------------------
# Generate Predictions
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("GENERATING PREDICTIONS")
println("="^60)

# Predictions on extended timespan
X_pred_hb = predict_ude(nn_hb, p_hb, st_hb, solver_hb, tspan_pred, t_pred)
X_pred_rs = predict_ude(nn_rs, p_rs, st_rs, solver_rs, tspan_pred, t_pred)

# Calculate errors
train_idx = 1:40
test_idx = 41:80

# Training errors
train_error_hb = mean(abs2, X_true_full[:, train_idx] - X_pred_hb[:, train_idx])
train_error_rs = mean(abs2, X_true_full[:, train_idx] - X_pred_rs[:, train_idx])

# Extrapolation errors
extrap_error_hb = mean(abs2, X_true_full[:, test_idx] - X_pred_hb[:, test_idx])
extrap_error_rs = mean(abs2, X_true_full[:, test_idx] - X_pred_rs[:, test_idx])

println("\n📊 PREDICTION ERRORS:")
println("  Training region (t=0 to 3):")
println("    Hyperband MSE: $(train_error_hb)")
println("    Random Search MSE: $(train_error_rs)")
println("    Improvement: $(round(train_error_rs/train_error_hb, digits=2))x")
println("\n  Extrapolation region (t=3 to 6):")
println("    Hyperband MSE: $(extrap_error_hb)")
println("    Random Search MSE: $(extrap_error_rs)")
println("    Improvement: $(round(extrap_error_rs/extrap_error_hb, digits=2))x")

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("CREATING VISUALIZATIONS")
println("="^60)

# Plot 1: Phase portraits
p1 = plot(title="Phase Portrait Comparison", 
         xlabel="Prey (x)", ylabel="Predator (y)",
         legend=:topright, size=(600, 500))

# Ground truth
plot!(p1, X_true_full[1, :], X_true_full[2, :], 
      label="Ground Truth", color=:black, linewidth=2, linestyle=:solid)

# Hyperband prediction
plot!(p1, X_pred_hb[1, :], X_pred_hb[2, :], 
      label="Hyperband UDE", color=:blue, linewidth=2, linestyle=:dash)

# Random Search prediction  
plot!(p1, X_pred_rs[1, :], X_pred_rs[2, :], 
      label="Random Search UDE", color=:red, linewidth=2, linestyle=:dot)

# Mark training/extrapolation boundary
scatter!(p1, [X_true_full[1, 40]], [X_true_full[2, 40]], 
         label="Training End", color=:green, markersize=8, marker=:star)

# Plot 2: Time series - Prey
p2 = plot(title="Prey Population Over Time", 
         xlabel="Time", ylabel="Population",
         legend=:topright, size=(600, 400))

# Add vertical line for training boundary
vline!(p2, [3.0], color=:gray, linestyle=:dash, linewidth=1, label="Training|Extrapolation")

# Ground truth
plot!(p2, t_pred, X_true_full[1, :], 
      label="Ground Truth", color=:black, linewidth=2)

# Hyperband
plot!(p2, t_pred, X_pred_hb[1, :], 
      label="Hyperband", color=:blue, linewidth=2, linestyle=:dash)

# Random Search
plot!(p2, t_pred, X_pred_rs[1, :], 
      label="Random Search", color=:red, linewidth=2, linestyle=:dot)

# Plot 3: Time series - Predator
p3 = plot(title="Predator Population Over Time", 
         xlabel="Time", ylabel="Population",
         legend=:topright, size=(600, 400))

vline!(p3, [3.0], color=:gray, linestyle=:dash, linewidth=1, label="Training|Extrapolation")

plot!(p3, t_pred, X_true_full[2, :], 
      label="Ground Truth", color=:black, linewidth=2)

plot!(p3, t_pred, X_pred_hb[2, :], 
      label="Hyperband", color=:blue, linewidth=2, linestyle=:dash)

plot!(p3, t_pred, X_pred_rs[2, :], 
      label="Random Search", color=:red, linewidth=2, linestyle=:dot)

# Plot 4: Absolute errors over time
p4 = plot(title="Prediction Error Over Time", 
         xlabel="Time", ylabel="Absolute Error",
         legend=:topleft, size=(600, 400), yaxis=:log)

vline!(p4, [3.0], color=:gray, linestyle=:dash, linewidth=1, label="Training|Extrapolation")

# Calculate point-wise errors
error_hb = vec(sqrt.(sum(abs2.(X_true_full - X_pred_hb), dims=1)))
error_rs = vec(sqrt.(sum(abs2.(X_true_full - X_pred_rs), dims=1)))

plot!(p4, t_pred, error_hb, 
      label="Hyperband", color=:blue, linewidth=2)

plot!(p4, t_pred, error_rs, 
      label="Random Search", color=:red, linewidth=2)

# Combine all plots
combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
display(combined)
savefig(combined, "lotka_volterra_comparison.png")

# Plot 5: Training convergence comparison
p5 = plot(title="Training Loss Convergence", 
         xlabel="Iteration", ylabel="Loss",
         legend=:topright, size=(600, 400), yaxis=:log)

plot!(p5, losses_hb[1:10:end], 
      label="Hyperband Config", color=:blue, linewidth=2)

plot!(p5, losses_rs[1:10:end], 
      label="Random Search Config", color=:red, linewidth=2)

display(p5)
savefig(p5, "training_convergence.png")

# -----------------------------------------------------------------------------
# Analyze Missing Physics Recovery
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("MISSING PHYSICS RECOVERY ANALYSIS")
println("="^60)

# The true missing term is γ*x*y where γ=0.8
function true_missing_term(x, y, t)
    return p_true[3] * x * y  # γ * x * y
end

# Sample points for comparison
test_points = [(x, y, t) for x in 0.5:0.5:2.0 for y in 0.5:0.5:5.0 for t in 0:0.5:3.0]

# Evaluate neural networks
missing_true = [true_missing_term(x, y, t) for (x, y, t) in test_points]
missing_hb = [nn_hb([x, y, t], p_hb, st_hb)[1][1] for (x, y, t) in test_points]
missing_rs = [nn_rs([x, y, t], p_rs, st_rs)[1][1] for (x, y, t) in test_points]

# Calculate R² scores
ss_tot = sum((missing_true .- mean(missing_true)).^2)
ss_res_hb = sum((missing_true - missing_hb).^2)
ss_res_rs = sum((missing_true - missing_rs).^2)

r2_hb = 1 - ss_res_hb/ss_tot
r2_rs = 1 - ss_res_rs/ss_tot

println("\nMissing Term Recovery (γ*x*y where γ=0.8):")
println("  Hyperband R² score: $(round(r2_hb, digits=4))")
println("  Random Search R² score: $(round(r2_rs, digits=4))")

# Plot missing physics comparison
p6 = scatter(missing_true, missing_hb, 
            label="Hyperband", color=:blue, alpha=0.6,
            xlabel="True γxy", ylabel="Predicted γxy",
            title="Missing Physics Recovery", legend=:topleft)
scatter!(p6, missing_true, missing_rs, 
         label="Random Search", color=:red, alpha=0.6)
plot!(p6, [minimum(missing_true), maximum(missing_true)], 
      [minimum(missing_true), maximum(missing_true)], 
      label="Perfect Recovery", color=:black, linestyle=:dash)

display(p6)
savefig(p6, "missing_physics_recovery.png")

println("\n✅ Analysis complete! Plots saved to:")
println("  - lotka_volterra_comparison.png")
println("  - training_convergence.png") 
println("  - missing_physics_recovery.png")

# -----------------------------------------------------------------------------
# Summary Table
# -----------------------------------------------------------------------------

using DataFrames

summary = DataFrame(
    Metric = [
        "Training Loss",
        "Training MSE", 
        "Extrapolation MSE",
        "Missing Physics R²",
        "Network Depth",
        "Learning Rate"
    ],
    Hyperband = [
        losses_hb[end],
        train_error_hb,
        extrap_error_hb,
        r2_hb,
        hyperband_config[:n_layers],
        hyperband_config[:learning_rate]
    ],
    RandomSearch = [
        losses_rs[end],
        train_error_rs,
        extrap_error_rs,
        r2_rs,
        random_config[:n_layers],
        random_config[:learning_rate]
    ],
    Improvement = [
        losses_rs[end]/losses_hb[end],
        train_error_rs/train_error_hb,
        extrap_error_rs/extrap_error_hb,
        r2_hb/r2_rs,
        "-",
        "-"
    ]
)

println("\n" * "="^60)
println("FINAL SUMMARY")
println("="^60)
display(summary)