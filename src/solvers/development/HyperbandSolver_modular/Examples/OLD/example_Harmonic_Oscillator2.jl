using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux, ComponentArrays, Zygote
using Plots
using Statistics, Random
using StableRNGs
using LinearAlgebra
using Printf
using LineSearches

include("../HyperbandSolver/HyperbandSolver.jl")
using .HyperbandSolver

# Set random seed for reproducibility
Random.seed!(1234)

# =============================================================================
# SIMPLE HARMONIC OSCILLATOR WITH ANALYTICAL SOLUTION
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
    const x₀ = 1.0      # Initial position
    const v₀ = 0.0      # Initial velocity

    # Analytical solution
    function analytical_solution(t; ω=ω_true, x₀=x₀, v₀=v₀)
        A = x₀
        B = v₀/ω
        x = A * cos.(ω * t) + B * sin.(ω * t)
        v = -A * ω * sin.(ω * t) + B * ω * cos.(ω * t)
        return vcat(x', v')
    end

    # True ODE system
    function harmonic_oscillator!(du, u, p, t)
        x, v = u
        ω = p[1]
        du[1] = v
        du[2] = -ω^2 * x
    end

    # Generate training data
    tspan = (0.0, 4π)  # Two full periods
    t_train = range(tspan[1], tspan[2], length=50)
    u0 = [x₀, v₀]

    # Solve true system
    prob_true = ODEProblem(harmonic_oscillator!, u0, tspan, [ω_true])
    sol_true = solve(prob_true, Tsit5(), saveat=t_train, abstol=1e-10, reltol=1e-10)

    # Add noise to create training data
    X_true = Array(sol_true)
    noise_level = 0.01  # Reduced noise for better training
    rng = StableRNGs.StableRNG(1234)
    X_noisy = X_true .+ noise_level * randn(rng, size(X_true))

    println("Generated training data: $(size(X_noisy, 2)) points with $(100*noise_level)% noise")

# =============================================================================
# UDE MODEL WITH PARTIAL PHYSICS
# =============================================================================

    """
    Universal Differential Equation for harmonic oscillator.
    We assume we know dx/dt = v, but we approximate dv/dt with a neural network.
    """
    function create_ude_model(n_hidden, n_layers, activation)
        # Ensure integer inputs
        n_hidden = round(Int, n_hidden)
        n_layers = round(Int, n_layers)
        
        # Build neural network architecture
        layers = []
        
        # Input layer
        push!(layers, Dense(2, n_hidden, activation))  # Takes [x, v] as input
        
        # Hidden layers
        for _ in 2:n_layers
            push!(layers, Dense(n_hidden, n_hidden, activation))
        end
        
        # Output layer - single output for dv/dt
        push!(layers, Dense(n_hidden, 1))
        
        return Chain(layers...)
    end

    # Better weight initialization based on expected output scale
    function initialize_nn_parameters(NN, rng)
        # Initialize with smaller weights for stability
        ps, st = Lux.setup(rng, NN)
        
        # Convert to Float64 and scale weights
        function scale_params(p)
            if p isa AbstractArray
                # Scale based on expected acceleration magnitude (~ω²*amplitude)
                return Float64.(p) * 0.1
            else
                return Float64(p)
            end
        end
        
        ps_scaled = ComponentVector(ps) |> x -> map(scale_params, x)
        
        return ComponentVector{Float64}(ps_scaled), st
    end

# =============================================================================
# HYPERPARAMETER SPACE DEFINITION (ADJUSTED)
# =============================================================================

    const HYPERPARAMETER_SPACE = (
        # Neural network architecture (smaller networks)
        n_hidden = (4, 16),            # Reduced hidden units
        n_layers = (2, 3),             # Fewer layers
        activation = (1, 2),           # 1=tanh, 2=sigmoid (removed relu)
        
        # ADAM optimizer parameters (wider range)
        adam_lr = (1e-3, 5e-1),        # Higher learning rates
        adam_beta1 = (0.8, 0.95),      # Adjusted beta1
        adam_beta2 = (0.9, 0.999),     # Adjusted beta2
        adam_iterations = (200, 2000), # More iterations
        
        # LBFGS optimizer parameters
        lbfgs_m = (3, 10),             # Reduced history size
        lbfgs_iterations = (100, 500), # More iterations
        linesearch = (1, 2)            # 1=BackTracking, 2=StrongWolfe
    )

    function sample_hyperparameters()
        hp = ComponentVector(
            n_hidden = rand(HYPERPARAMETER_SPACE.n_hidden[1]:HYPERPARAMETER_SPACE.n_hidden[2]),
            n_layers = rand(HYPERPARAMETER_SPACE.n_layers[1]:HYPERPARAMETER_SPACE.n_layers[2]),
            activation = rand(HYPERPARAMETER_SPACE.activation[1]:HYPERPARAMETER_SPACE.activation[2]),
            adam_lr = HYPERPARAMETER_SPACE.adam_lr[1] * 10^(rand() * log10(HYPERPARAMETER_SPACE.adam_lr[2]/HYPERPARAMETER_SPACE.adam_lr[1])),
            adam_beta1 = HYPERPARAMETER_SPACE.adam_beta1[1] + rand() * (HYPERPARAMETER_SPACE.adam_beta1[2] - HYPERPARAMETER_SPACE.adam_beta1[1]),
            adam_beta2 = HYPERPARAMETER_SPACE.adam_beta2[1] + rand() * (HYPERPARAMETER_SPACE.adam_beta2[2] - HYPERPARAMETER_SPACE.adam_beta2[1]),
            adam_iterations = rand(HYPERPARAMETER_SPACE.adam_iterations[1]:HYPERPARAMETER_SPACE.adam_iterations[2]),
            lbfgs_m = rand(HYPERPARAMETER_SPACE.lbfgs_m[1]:HYPERPARAMETER_SPACE.lbfgs_m[2]),
            lbfgs_iterations = rand(HYPERPARAMETER_SPACE.lbfgs_iterations[1]:HYPERPARAMETER_SPACE.lbfgs_iterations[2]),
            linesearch = rand(HYPERPARAMETER_SPACE.linesearch[1]:HYPERPARAMETER_SPACE.linesearch[2])
        )
        return hp
    end

# =============================================================================
# TRAINING FUNCTION WITH GIVEN HYPERPARAMETERS (FIXED)
# =============================================================================

    function train_ude_with_hyperparameters(hp, resource_budget; verbose=false)
        try
            # Select activation function (avoid ReLU for this problem)
            activation = if hp.activation == 1
                tanh
            else
                sigmoid
            end
            
            # Create neural network
            NN = create_ude_model(hp.n_hidden, hp.n_layers, activation)
            
            # Initialize parameters with better scaling
            rng = StableRNGs.StableRNG(1234)
            p_nn, st_nn = initialize_nn_parameters(NN, rng)
            
            # Create UDE problem with closure
            function ude_dynamics_closure!(du, u, p, t)
                x, v = u
                
                # Known physics: dx/dt = v
                du[1] = v
                
                # Unknown physics approximated by NN: dv/dt = NN([x, v])
                nn_input = [x, v]
                dv_pred, _ = Lux.apply(NN, nn_input, p, st_nn)
                du[2] = dv_pred[1]
            end
            
            prob_ude = ODEProblem(ude_dynamics_closure!, u0, tspan, p_nn)
            
            # Prediction function with better error handling
            function predict(θ)
                try
                    # Use simpler sensitivity algorithm for stability
                    sol = solve(prob_ude, Tsit5(), 
                            p=θ, 
                            saveat=t_train,
                            sensealg=ForwardDiffSensitivity(),  # More stable
                            #sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                            abstol=1e-6, 
                            reltol=1e-6,
                            maxiters=1e5)
                    
                    if sol.retcode == :Success
                        return Array(sol)
                    else
                        return nothing
                    end
                catch e
                    if verbose
                        println("    Prediction failed: $(typeof(e))")
                    end
                    return nothing
                end
            end
            
            # Loss function with regularization
            function loss(θ)
                pred = predict(θ)
                if isnothing(pred)
                    return Inf
                end
                
                # MSE loss with small L2 regularization
                mse_loss = mean(abs2, X_noisy .- pred)
                reg_loss = 1e-6 * sum(abs2, θ)  # Small regularization
                
                return mse_loss + reg_loss
            end
            
            # Scale iterations based on resource budget
            adam_iters = min(hp.adam_iterations, round(Int, resource_budget * 0.8))
            lbfgs_iters = min(hp.lbfgs_iterations, round(Int, resource_budget * 0.2))
            
            # Early stopping if resource budget is too small
            if adam_iters < 10
                return (loss = Inf, analytical_error = Inf, parameters = nothing, 
                    iterations = 0, hyperparameters = hp, NN = nothing, st = nothing)
            end
            
            losses = Float64[]
            
            # Callback with early stopping
            callback = function(state, l)
                push!(losses, l)
                if verbose && length(losses) % 100 == 0
                    println("  Iteration $(length(losses)): loss = $l")
                end
                # Early stopping if loss is NaN or too large
                if isnan(l) || l > 1e10
                    return true
                end
                return false
            end
            
            # Stage 1: ADAM optimization
            adtype = Optimization.AutoZygote()
            optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
            optprob = Optimization.OptimizationProblem(optf, p_nn)
            #optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_nn))
            
            adam_opt = OptimizationOptimisers.Adam(hp.adam_lr, (hp.adam_beta1, hp.adam_beta2))
            res1 = solve(optprob, adam_opt, callback=callback, maxiters=adam_iters)
            
            # Skip LBFGS if ADAM failed
            if res1.objective == Inf || isnan(res1.objective)
                return (loss = Inf, analytical_error = Inf, parameters = nothing, 
                    iterations = length(losses), hyperparameters = hp, NN = nothing, st = nothing)
            end
            
            # Stage 2: LBFGS refinement (optional, only if ADAM succeeded)
            if lbfgs_iters > 0
                linesearch_alg = if hp.linesearch == 1
                    LineSearches.BackTracking()
                else
                    LineSearches.StrongWolfe()
                end
                
                optprob2 = Optimization.OptimizationProblem(optf, res1.u)
                lbfgs_opt = OptimizationOptimJL.LBFGS(m=hp.lbfgs_m, linesearch=linesearch_alg)
                
                try
                    res2 = solve(optprob2, lbfgs_opt, callback=callback, maxiters=lbfgs_iters)
                    final_params = res2.u
                    final_loss = res2.objective
                catch
                    # If LBFGS fails, keep ADAM results
                    final_params = res1.u
                    final_loss = res1.objective
                end
            else
                final_params = res1.u
                final_loss = res1.objective
            end
            
            # Compute prediction error on analytical solution
            pred_final = predict(final_params)
            if !isnothing(pred_final)
                X_analytical = analytical_solution(t_train)
                analytical_error = mean(abs2, X_analytical .- pred_final)
                
                return (
                    loss = final_loss,
                    analytical_error = analytical_error,
                    parameters = final_params,
                    iterations = length(losses),
                    hyperparameters = hp,
                    NN = NN,
                    st = st_nn
                )
            end
            
        catch e
            if verbose
                println("Training failed with error: $(typeof(e))")
                println("Error message: $e")
            end
        end
        
        return (loss = Inf, analytical_error = Inf, parameters = nothing, 
            iterations = 0, hyperparameters = hp, NN = nothing, st = nothing)
    end

# =============================================================================
# HYPERBAND OPTIMIZATION
# =============================================================================

    println("\n" * "="^60)
    println("HYPERBAND OPTIMIZATION FOR UDE HYPERPARAMETERS")
    println("="^60)

    # Objective function for Hyperband
    function hyperband_objective(config_vec, resource)
        # Convert vector to hyperparameter struct
        hp = ComponentVector(
            n_hidden = round(Int, config_vec[1]),
            n_layers = round(Int, config_vec[2]),
            activation = round(Int, config_vec[3]),
            adam_lr = config_vec[4],
            adam_beta1 = config_vec[5],
            adam_beta2 = config_vec[6],
            adam_iterations = round(Int, config_vec[7]),
            lbfgs_m = round(Int, config_vec[8]),
            lbfgs_iterations = round(Int, config_vec[9]),
            linesearch = round(Int, config_vec[10])
        )
        
        result = train_ude_with_hyperparameters(hp, resource, verbose=false)
        return result.loss
    end

    # Configuration generator for Hyperband
    function get_random_config()
        hp = sample_hyperparameters()
        return [
            Float64(hp.n_hidden),
            Float64(hp.n_layers),
            Float64(hp.activation),
            hp.adam_lr,
            hp.adam_beta1,
            hp.adam_beta2,
            Float64(hp.adam_iterations),
            Float64(hp.lbfgs_m),
            Float64(hp.lbfgs_iterations),
            Float64(hp.linesearch)
        ]
    end

    # Run Hyperband with reduced maximum resource
    println("\nStarting Hyperband search...")
    println("Hyperparameter space dimensions: 10")
    println("Maximum resource (iterations): 100000")

    best_config, best_loss = HyperbandSolver.hyperband(
        hyperband_objective,
        get_random_config,
        100000,  # Reduced maximum iterations
        η = 3
    )

    # Check if Hyperband found a valid solution
    if best_config === nothing || best_loss == Inf
        println("\n" * "="^60)
        println("HYPERBAND FAILED TO FIND VALID SOLUTION")
        println("="^60)
        println("All configurations returned Inf loss.")
        println("\nTrying a simple baseline configuration...")
        
        # Try a simple baseline configuration
        baseline_hp = ComponentVector(
            n_hidden = 8,
            n_layers = 2,
            activation = 1,  # tanh
            adam_lr = 0.01,
            adam_beta1 = 0.9,
            adam_beta2 = 0.999,
            adam_iterations = 1000,
            lbfgs_m = 5,
            lbfgs_iterations = 200,
            linesearch = 1  # BackTracking
        )
        
        println("\nBaseline configuration:")
        for (k, v) in pairs(baseline_hp)
            println("  $k: $v")
        end
        
        baseline_result = train_ude_with_hyperparameters(baseline_hp, 1200, verbose=true)
        
        if baseline_result.loss < Inf
            println("\nBaseline succeeded with loss: $(baseline_result.loss)")
            best_hp = baseline_hp
            best_loss = baseline_result.loss
            best_config = [Float64(v) for v in values(baseline_hp)]
        else
            error("Even baseline configuration failed. Check problem setup.")
        end
    else
        # Convert best config back to hyperparameters
        best_hp = ComponentVector(
            n_hidden = round(Int, best_config[1]),
            n_layers = round(Int, best_config[2]),
            activation = round(Int, best_config[3]),
            adam_lr = best_config[4],
            adam_beta1 = best_config[5],
            adam_beta2 = best_config[6],
            adam_iterations = round(Int, best_config[7]),
            lbfgs_m = round(Int, best_config[8]),
            lbfgs_iterations = round(Int, best_config[9]),
            linesearch = round(Int, best_config[10])
        )
    end

    println("\n" * "="^60)
    println("BEST HYPERPARAMETERS FOUND")
    println("="^60)
    println("Neural Network Architecture:")
    println("  Hidden units: $(best_hp.n_hidden)")
    println("  Number of layers: $(best_hp.n_layers)")
    println("  Activation: $(best_hp.activation == 1 ? "tanh" : "sigmoid")")
    println("\nADAM Optimizer:")
    println("  Learning rate: $(Printf.@sprintf("%.6f", best_hp.adam_lr))")
    println("  Beta1: $(Printf.@sprintf("%.4f", best_hp.adam_beta1))")
    println("  Beta2: $(Printf.@sprintf("%.4f", best_hp.adam_beta2))")
    println("  Iterations: $(best_hp.adam_iterations)")
    println("\nLBFGS Optimizer:")
    println("  History size (m): $(best_hp.lbfgs_m)")
    println("  Iterations: $(best_hp.lbfgs_iterations)")
    println("  Line search: $(best_hp.linesearch == 1 ? "BackTracking" : "StrongWolfe")")
    println("\nBest loss achieved: $(Printf.@sprintf("%.6f", best_loss))")

# =============================================================================
# FINAL TRAINING WITH BEST HYPERPARAMETERS
# =============================================================================

    println("\n" * "="^60)
    println("FINAL TRAINING WITH BEST HYPERPARAMETERS")
    println("="^60)

    final_result = train_ude_with_hyperparameters(
        best_hp, 
        best_hp.adam_iterations + best_hp.lbfgs_iterations, 
        verbose=true
    )

    println("\nFinal Results:")
    println("  Training loss: $(Printf.@sprintf("%.6f", final_result.loss))")
    println("  Error vs analytical solution: $(Printf.@sprintf("%.6f", final_result.analytical_error))")
    println("  Total iterations: $(final_result.iterations)")

# =============================================================================
# COMPARISON WITH ANALYTICAL SOLUTION AND VISUALIZATION
# =============================================================================

    if !isnothing(final_result.NN) && !isnothing(final_result.st)
        # Use the trained model for predictions
        NN_final = final_result.NN
        st_final = final_result.st
        
        function ude_dynamics_final!(du, u, p, t)
            x, v = u
            du[1] = v
            nn_input = [x, v]
            dv_pred, _ = Lux.apply(NN_final, nn_input, p, st_final)
            du[2] = dv_pred[1]
        end
        
        prob_ude_final = ODEProblem(ude_dynamics_final!, u0, tspan, final_result.parameters)
        
        # Extended time for extrapolation test
        t_test = range(0.0, 6π, length=200)
        sol_ude = solve(prob_ude_final, Tsit5(), saveat=t_test, abstol=1e-8, reltol=1e-8)
        
        if sol_ude.retcode == :Success
            X_ude = Array(sol_ude)
            
            # Analytical solution
            X_analytical = analytical_solution(t_test)
            
            # Compute errors
            position_error = mean(abs2, X_ude[1, :] .- X_analytical[1, :])
            velocity_error = mean(abs2, X_ude[2, :] .- X_analytical[2, :])
            total_error = mean(abs2, X_ude .- X_analytical)
            
            println("\n" * "="^60)
            println("COMPARISON WITH ANALYTICAL SOLUTION")
            println("="^60)
            println("Mean Squared Errors:")
            println("  Position (x): $(Printf.@sprintf("%.6f", position_error))")
            println("  Velocity (v): $(Printf.@sprintf("%.6f", velocity_error))")
            println("  Total: $(Printf.@sprintf("%.6f", total_error))")
            
            # Relative errors
            rel_position_error = sqrt(position_error) / sqrt(mean(abs2, X_analytical[1, :]))
            rel_velocity_error = sqrt(velocity_error) / sqrt(mean(abs2, X_analytical[2, :]))
            rel_total_error = sqrt(total_error) / sqrt(mean(abs2, X_analytical))
            
            println("\nRelative Errors:")
            println("  Position (x): $(Printf.@sprintf("%.2f", 100*rel_position_error))%")
            println("  Velocity (v): $(Printf.@sprintf("%.2f", 100*rel_velocity_error))%")
            println("  Total: $(Printf.@sprintf("%.2f", 100*rel_total_error))%")
            
            # =============================================================================
            # VISUALIZATION
            # =============================================================================
            
            # Plot comparison
            p1 = plot(t_test, X_analytical[1, :], label="Analytical x(t)", color=:black, linewidth=2)
            plot!(p1, t_test, X_ude[1, :], label="UDE x(t)", color=:blue, linestyle=:dash, linewidth=2)
            scatter!(p1, t_train, X_noisy[1, :], label="Training data", color=:red, alpha=0.5, markersize=2)
            xlabel!(p1, "Time")
            ylabel!(p1, "Position")
            title!(p1, "Position Comparison")
            
            p2 = plot(t_test, X_analytical[2, :], label="Analytical v(t)", color=:black, linewidth=2)
            plot!(p2, t_test, X_ude[2, :], label="UDE v(t)", color=:blue, linestyle=:dash, linewidth=2)
            scatter!(p2, t_train, X_noisy[2, :], label="Training data", color=:red, alpha=0.5, markersize=2)
            xlabel!(p2, "Time")
            ylabel!(p2, "Velocity")
            title!(p2, "Velocity Comparison")
            
            # Phase portrait
            p3 = plot(X_analytical[1, :], X_analytical[2, :], label="Analytical", color=:black, linewidth=2)
            plot!(p3, X_ude[1, :], X_ude[2, :], label="UDE", color=:blue, linestyle=:dash, linewidth=2)
            scatter!(p3, X_noisy[1, :], X_noisy[2, :], label="Training data", color=:red, alpha=0.5, markersize=2)
            xlabel!(p3, "Position")
            ylabel!(p3, "Velocity")
            title!(p3, "Phase Portrait")
            
            # Error over time
            p4 = plot(t_test, abs.(X_ude[1, :] .- X_analytical[1, :]), label="Position error", color=:red)
            plot!(p4, t_test, abs.(X_ude[2, :] .- X_analytical[2, :]), label="Velocity error", color=:blue)
            xlabel!(p4, "Time")
            ylabel!(p4, "Absolute Error")
            title!(p4, "Prediction Error Over Time")
            vline!(p4, [tspan[2]], label="End of training data", color=:gray, linestyle=:dash)
            
            final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
            display(final_plot)
        else
            println("\nFinal prediction failed - unable to generate visualizations")
        end
    else
        println("\nTraining failed - unable to generate visualizations")
    end

    println("\n" * "="^60)
    println("OPTIMIZATION COMPLETE")
    println("="^60)