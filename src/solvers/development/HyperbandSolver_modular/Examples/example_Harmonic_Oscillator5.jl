# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux, ComponentArrays, Zygote
using Plots
using Statistics, Random
using DataFrames
using BenchmarkTools
using LinearAlgebra
using Printf

include("../HyperbandSolver/HyperbandSolver.jl")
using .HyperbandSolver

# Set random seed
Random.seed!(1234)
rng = Random.default_rng()

# =============================================================================
# HARMONIC OSCILLATOR PROBLEM SETUP
# =============================================================================

"""
Simple Harmonic Oscillator ODE:
    d²x/dt² + ω²x = 0
    
Rewritten as system:
    dx/dt = v
    dv/dt = -ω²x
    
Analytical solution:
    x(t) = A*cos(ωt) + B*sin(ωt)
    v(t) = -A*ω*sin(ωt) + B*ω*cos(ωt)
    
With initial conditions x(0) = x₀, v(0) = v₀:
    A = x₀, B = v₀/ω
"""

# True system parameters
const ω_true = 2.0  # Natural frequency

# True ODE system
function harmonic_oscillator!(du, u, p, t)
    x, v = u
    ω = p[1]
    du[1] = v
    du[2] = -ω^2 * x
end

# Generate training data
tspan = (0.0, 4π)
const t_train = range(tspan[1], tspan[2], length=50)
u0 = [1.0, 0.0]

prob_true = ODEProblem(harmonic_oscillator!, u0, tspan, [ω_true])
sol_true = solve(prob_true, Tsit5(), saveat=t_train, abstol=1e-10, reltol=1e-10)

X_true = Array(sol_true)
noise_level = 0.01
X_noisy = X_true .+ noise_level * randn(rng, size(X_true))

# Plot the noisy data
plot(t_train, X_noisy[1, :], color=:red, label="Noisy x(t)", linewidth=2)
plot!(t_train, X_noisy[2, :], color=:blue, label="Noisy v(t)", linewidth=2)
scatter!(t_train, X_noisy[1, :], color=:red, alpha=0.6, markersize=3, label=nothing)
scatter!(t_train, X_noisy[2, :], color=:blue, alpha=0.6, markersize=3, label=nothing)

# Add the true solution for comparison
plot!(sol_true, alpha=0.75, color=:black, linestyle=:dash, label=["True x(t)" "True v(t)"])

xlabel!("Time")
ylabel!("State Variables")
title!("Harmonic Oscillator: True vs Noisy Data")

println("Generated training data: $(size(X_noisy, 2)) points with $(100*noise_level)% noise")

# =============================================================================
# UDE MODEL (FOLLOWING SCIML APPROACH)
# =============================================================================
"""
Universal Differential Equation for harmonic oscillator.
We assume we know dx/dt = v, but we approximate dv/dt with a neural network.
"""
function create_ude_model(config)
    # Extract hyperparameters
    hidden_dim = config[:hidden_dim]
    n_layers = config[:n_layers]
    activation = config[:activation]
    
    # Build the neural network layers
    layers = []
    push!(layers, Dense(2, hidden_dim, activation))  # Input: [x, v]
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim, hidden_dim, activation))
    end
    push!(layers, Dense(hidden_dim, 1))  # Output: dv/dt
    
    # Create the model
    nn_model = Lux.Chain(layers...)
    
    # Initialize parameters
    p_model, st = Lux.setup(rng, nn_model)
    p_model = ComponentArray{Float64}(p_model)  # Force Float64
    
    # Define the UDE dynamics
    function ude_dynamics!(du, u, p, t)
        x, v = u
        
        # Known physics
        du[1] = v
        
        # Unknown physics (NN)
        nn_input = Float64[x, v]
        dv_pred, _ = Lux.apply(nn_model, nn_input, p, st)
        du[2] = dv_pred[1]
    end
    
    # Create ODE problem
    prob_nn = ODEProblem(ude_dynamics!, u0, tspan, p_model)
    
    return prob_nn, nn_model, p_model, st
end

# =============================================================================
# IMPROVED TRAINING FUNCTION
# =============================================================================

function train_ude_model(config::Dict, resource::Int)
    try
        # Skip if resource is too low (minimum meaningful training)
        if resource < 20
            return 1e8
        end
        
        # Create UDE model
        prob_nn, nn_model, p_init, st = create_ude_model(config)
        
        # Extract training hyperparameters
        adam_lr = config[:adam_lr]
        split_ratio = config[:adam_lbfgs_split]
        lbfgs_max_iters = config[:lbfgs_max_iters]

        # Ensure minimum iterations
        adam_iters = max(10, round(Int, resource * split_ratio))
        lbfgs_iters = max(0, min(lbfgs_max_iters, round(Int, resource * (1 - split_ratio))))
        
        # Use simpler, more robust solver for initial training
        solver = Tsit5()  # Fixed robust solver
        
        # Simpler prediction function without sensitivity
        function predict(θ)
            try
                _prob = remake(prob_nn, p=θ)
                sol = solve(_prob, solver, saveat=t_train,
                           abstol=1e-6, reltol=1e-5,
                           maxiters=10000)
                
                if sol.retcode == :Success
                    return Array(sol)
                else
                    return nothing
                end
            catch
                return nothing
            end
        end
        
        # Loss function with regularization
        function loss(θ)
            pred = predict(θ)
            if isnothing(pred) || any(isnan, pred) || any(isinf, pred)
                return 1e8
            end
            
            # MSE loss with L2 regularization
            mse_loss = mean(abs2, X_noisy .- pred)
            reg_loss = 1e-6 * sum(abs2, θ)  # Small L2 regularization
            total_loss = mse_loss + reg_loss
            
            if isnan(total_loss) || isinf(total_loss) || total_loss > 1e6
                return 1e8
            end
            
            return total_loss
        end
        
        # Test initial parameters
        initial_loss = loss(p_init)
        if initial_loss >= 1e8
            # Try with smaller initial parameters
            p_init = p_init .* 0.1
            initial_loss = loss(p_init)
            if initial_loss >= 1e8
                return 1e8
            end
        end
        
        # Simple callback without printing for speed
        losses = Float64[]
        callback = function (p, l)
            push!(losses, l)
            return false
        end
        
        # STAGE 1: ADAM OPTIMIZATION
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
        
        adam_opt = OptimizationOptimisers.Adam(adam_lr)
        
        local res1
        try
            res1 = solve(optprob, adam_opt, callback=callback, maxiters=adam_iters)
        catch e
            return initial_loss  # Return initial loss if optimization fails
        end
        
        final_loss = res1.objective
        
        # STAGE 2: LBFGS REFINEMENT (only if ADAM was successful)
        if lbfgs_iters > 0 && final_loss < 100 && resource > 50
            optprob2 = Optimization.OptimizationProblem(optf, res1.u)
            
            try
                res2 = solve(optprob2, 
                            OptimizationOptimJL.LBFGS(),
                            callback = callback,
                            maxiters = lbfgs_iters)
                
                if res2.objective < final_loss && !isnan(res2.objective)
                    final_loss = res2.objective
                end
            catch
                # Keep ADAM results if LBFGS fails
            end
        end
        
        return final_loss
        
    catch e
        return 1e8
    end
end

# =============================================================================
# SIMPLIFIED CONFIGURATION SPACE
# =============================================================================

function get_ude_config()
    Dict(
        :hidden_dim => rand([16, 32, 64]),           # Good range
        :n_layers => rand(2:4),                      # Not too deep
        :activation => tanh,                         # Fixed, reliable activation
        :adam_lr => 10.0^(-rand() * 2 - 2),         # 10^-4 to 10^-2
        :lbfgs_max_iters => rand([50, 100, 200]),   # Reasonable range
        :adam_lbfgs_split => rand([0.7, 0.8, 0.9]), # How to split resources
        :solver => Tsit5()                          # Fixed robust solver
    )
end

# =============================================================================
# OBJECTIVE FUNCTION FOR HYPERBAND (MISSING IN YOUR CODE)
# =============================================================================

# Counter for function evaluations
eval_count = 0

# This is the objective function that Hyperband calls
function counted_ude_objective(config::Dict, resource::Int)
    global eval_count
    eval_count += 1
    return train_ude_model(config, resource)
end

# =============================================================================
# MULTI-RUN HYPERBAND WITH STATISTICS
# =============================================================================

function run_hyperband_experiments(n_runs=3)
    results = []
    best_overall_config = nothing
    best_overall_loss = Inf
    
    for run in 1:n_runs
        println("\n" * "="^60)
        println("HYPERBAND RUN $run/$n_runs")
        println("="^60)
        
        # Reset counter
        global eval_count = 0
        
        t_start = time()
        config, loss = HyperbandSolver.hyperband(
            counted_ude_objective,
            get_ude_config,
            1000,  # Increased from 500 but not too high
            η = 3
        )
        run_time = time() - t_start
        
        push!(results, (
            run = run,
            config = config,
            loss = loss,
            time = run_time,
            evals = eval_count
        ))
        
        if loss < best_overall_loss
            best_overall_loss = loss
            best_overall_config = config
        end
        
        println("Run $run completed: loss = $loss, time = $(round(run_time, digits=2))s, evals = $eval_count")
    end
    
    return results, best_overall_config, best_overall_loss
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Run the multi-run experiment
println("\n" * "="^60)
println("MULTI-RUN HYPERBAND OPTIMIZATION")
println("="^60)

results, best_config, best_loss = run_hyperband_experiments(3)

# Analyze results
valid_results = filter(r -> r.loss < 1e8, results)

if length(valid_results) > 0
    println("\n" * "="^60)
    println("OPTIMIZATION SUMMARY")
    println("="^60)
    println("Valid runs: $(length(valid_results))/$(length(results))")
    
    losses = [r.loss for r in valid_results]
    times = [r.time for r in valid_results]
    evals = [r.evals for r in valid_results]
    
    println("Loss statistics:")
    println("  Best: $(minimum(losses))")
    println("  Mean: $(mean(losses))")
    println("  Std: $(std(losses))")
    
    println("Time statistics:")
    println("  Mean: $(round(mean(times), digits=2))s")
    println("  Total: $(round(sum(times), digits=2))s")
    
    println("Evaluations:")
    println("  Mean per run: $(round(mean(evals), digits=0))")
    println("  Total: $(sum(evals))")
    
    # Display best configuration
    if best_config !== nothing && best_loss < 1e8
        println("\n" * "="^60)
        println("BEST CONFIGURATION FOUND")
        println("="^60)
        println("Best loss: $(Printf.@sprintf("%.6f", best_loss))")
        for (key, value) in best_config
            println("  $key: $value")
        end
        
        # Final training with best config
        println("\nFinal training with extended resources...")
        final_loss = train_ude_model(best_config, 2000)
        println("Final loss: $(Printf.@sprintf("%.6f", final_loss))")
        
        # Visualize the best model
        println("\n" * "="^60)
        println("VISUALIZING BEST MODEL")
        println("="^60)
        
        # Create and train the best model for visualization
        prob_nn, nn_model, p_best, st = create_ude_model(best_config)
        
        # Quick re-train for visualization
        function predict_best(θ)
            _prob = remake(prob_nn, p=θ)
            sol = solve(_prob, Tsit5(), saveat=t_train, abstol=1e-6, reltol=1e-5)
            return Array(sol)
        end
        
        # Generate prediction
        X_pred = predict_best(p_best)
        
        # Plot comparison
        p_comparison = plot(t_train, X_true[1, :], label="True x(t)", linewidth=2, color=:black)
        plot!(t_train, X_pred[1, :], label="Predicted x(t)", linewidth=2, color=:red, linestyle=:dash)
        scatter!(t_train, X_noisy[1, :], label="Noisy data", alpha=0.5, color=:gray, markersize=2)
        
        plot!(t_train, X_true[2, :], label="True v(t)", linewidth=2, color=:blue)
        plot!(t_train, X_pred[2, :], label="Predicted v(t)", linewidth=2, color=:orange, linestyle=:dash)
        scatter!(t_train, X_noisy[2, :], label=nothing, alpha=0.5, color=:gray, markersize=2)
        
        xlabel!("Time")
        ylabel!("State")
        title!("UDE Model Performance (Best Hyperparameters)")
        
        display(p_comparison)
    end
else
    println("\nAll runs failed. Trying a baseline configuration...")
    
    # Try a known good baseline
    baseline_config = Dict(
        :hidden_dim => 32,
        :n_layers => 3,
        :activation => tanh,
        :adam_lr => 0.001,
        :lbfgs_max_iters => 100,
        :adam_lbfgs_split => 0.8,
        :solver => Tsit5()
    )
    
    baseline_loss = train_ude_model(baseline_config, 1000)
    println("Baseline loss: $baseline_loss")
    
    if baseline_loss < 1e8
        println("Baseline configuration works! Using it for visualization...")
        best_config = baseline_config
        best_loss = baseline_loss
    else
        println("Even baseline configuration failed. Check your problem setup.")
    end
end

println("\n" * "="^60)
println("OPTIMIZATION COMPLETE")
println("="^60)