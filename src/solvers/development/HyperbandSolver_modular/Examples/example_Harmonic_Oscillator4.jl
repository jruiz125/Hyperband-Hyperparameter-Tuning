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
# TRAINING FUNCTION
# =============================================================================

    """
    Train UDE model with given configuration and resource budget
    """
    function train_ude_model(config::Dict, resource::Int)
        try
            # Create UDE model
            prob_nn, nn_model, p_init, st = create_ude_model(config)
            
            # Extract training hyperparameters
            adam_lr = config[:adam_lr]
            split_ratio = config[:adam_lbfgs_split]
            lbfgs_max_iters = config[:lbfgs_max_iters]

            adam_iters = max(1, round(Int, resource * split_ratio))
            lbfgs_iters = max(5, min(lbfgs_max_iters, round(Int, resource * (1 - split_ratio))))
            solver = config[:solver]

            if adam_iters < 1
                return 1e8
            end
            
            # Define the prediction function WITH sensitivity algorithm
            function predict(θ)
                _prob = remake(prob_nn, p=θ)
                sol = solve(_prob, solver, saveat=t_train,  # Store result in sol
                            abstol=1e-4, reltol=1e-3,
                            sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
                
                if sol.retcode == :Success
                    return Array(sol)
                else
                    return nothing
                end
            end
            
            # Loss function
            function loss(θ)
                pred = predict(θ)
                if isnothing(pred)
                    return 1e8
                end
                
                mse_loss = mean(abs2, X_noisy .- pred)
                
                if isnan(mse_loss) || isinf(mse_loss)
                    return 1e8
                end
                
                return mse_loss
            end
            
            # Callback
            losses = Float64[]
            callback = function (p, l)
                push!(losses, l)
                if length(losses) % 10 == 0
                    println("    Iteration $(length(losses)): Loss = $l")
                end
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
                return 1e8
            end
            
            if res1.objective >= 1e8
                return res1.objective
            end
            
            # STAGE 2: LBFGS REFINEMENT
            final_loss = res1.objective
            
            if lbfgs_iters > 0 && final_loss < 100
                optprob2 = Optimization.OptimizationProblem(optf, res1.u)
                
                try
                    res2 = solve(optprob2, 
                                OptimizationOptimJL.LBFGS(),
                                callback = callback,
                                maxiters = lbfgs_iters)
                    
                    if res2.objective < final_loss
                        final_loss = res2.objective
                    end
                catch
                    # Keep ADAM results
                end
            end
            
            return final_loss
            
        catch e
            return 1e8
        end
    end

# =============================================================================
# HYPERPARAMETER CONFIGURATION SPACE
# =============================================================================
    # Configuration generator for Hyperband
    function get_ude_config()
        Dict(
            :hidden_dim => rand([8, 16, 32]),
            :n_layers => rand(2:5),
            :activation => rand([tanh, sigmoid]),
            :adam_lr => 10.0^(-rand() * 3 - 1),  # 10^-4 to 10^-1
            :lbfgs_max_iters => rand([10, 20, 50, 1000]),
            :adam_lbfgs_split => rand([0.7, 0.8, 0.9]),  # How to split resource between stages
            :solver => rand([Tsit5(), Vern7(), AutoTsit5(Rosenbrock23()), AutoTsit5(Rodas5())])
        )
    end
    # Objective function for Hyperband

    eval_count = 0 # Counter for function evaluations

    # This is the equivalent of hyperband_objective
    function counted_ude_objective(config::Dict, resource::Int)
        global eval_count
        eval_count += 1
        return train_ude_model(config, resource)
    end

# =============================================================================
# HYPERBAND OPTIMIZATION
# =============================================================================

    println("\n" * "="^60)
    println("HYPERBAND OPTIMIZATION (Simplified Pattern)")
    println("="^60)

    t_start = time()
    # train 
    best_config, best_loss = HyperbandSolver.hyperband(
        counted_ude_objective,
        get_ude_config,
        2000,
        η = 3
    )
    hb_time = time() - t_start

    if best_config !== nothing && best_loss < 1e8
        println("\n" * "="^60)
        println("BEST HYPERPARAMETERS FOUND")
        println("="^60)
        println("Total function evaluations: $eval_count")
        println("Best loss: $(Printf.@sprintf("%.6f", best_loss))")
        println("Best configuration:")
        for (key, value) in best_config
            if key == :activation || key == :solver
                println("  $key: $(string(value))")
            else
                println("  $key: $value")
            end
        end
        
        # Final training with more resources
        println("\n" * "="^60)
        println("FINAL TRAINING WITH BEST HYPERPARAMETERS")
        println("="^60)
        
        final_loss = train_ude_model(best_config, 1000)
        println("Final loss with extended training: $(Printf.@sprintf("%.6f", final_loss))")
        
        # Visualization using BEST hyperparameters
        if final_loss < 1e8
            println("\n" * "="^60)
            println("CREATING VISUALIZATION")
            println("="^60)
            
            # Create final model for visualization
            prob_nn_final, NN_final, ps_final, st_final = create_ude_model(best_config)
            
            # Train final model
            function predict_final(θ)
                sol = solve(prob_nn_final, best_config[:solver],  # Use best solver!
                        p = θ,
                        saveat = t_train,
                        abstol = 1e-6,
                        reltol = 1e-6)
                return Array(sol)
            end
            
            function loss_final(θ)
                pred = predict_final(θ)
                return mean(abs2, X_noisy .- pred)
            end
            
            # Quick training for visualization 
            adtype = Optimization.AutoZygote()
            optf = Optimization.OptimizationFunction((x, p) -> loss_final(x), adtype)
            optprob = Optimization.OptimizationProblem(optf, ps_final)

            # Use the OPTIMAL hyperparameters found by Hyperband
            visualization_resource = 400  # Resource budget for final training
            split_ratio = best_config[:adam_lbfgs_split]
            lbfgs_max_iters = best_config[:lbfgs_max_iters]

            # Calculate optimal iterations using best hyperparameters
            adam_iters_final = max(10, round(Int, visualization_resource * split_ratio))
            lbfgs_iters_final = max(5, min(lbfgs_max_iters, round(Int, visualization_resource * (1 - split_ratio))))

            # STAGE 1: ADAM with optimal learning rate and iterations
            adam_opt = OptimizationOptimisers.Adam(best_config[:adam_lr])
            res_final = solve(optprob, adam_opt, maxiters=adam_iters_final)  # Use optimal iterations!

            # STAGE 2: LBFGS refinement (if beneficial)
            if lbfgs_iters_final > 0 && res_final.objective < 100
                optprob2 = Optimization.OptimizationProblem(optf, res_final.u)
                
                try
                    res_lbfgs = solve(optprob2, 
                                    OptimizationOptimJL.LBFGS(),
                                    maxiters = lbfgs_iters_final)  # Use optimal LBFGS iterations!
                    
                    if res_lbfgs.objective < res_final.objective
                        res_final = res_lbfgs  # Use LBFGS result if better
                        println("LBFGS refinement improved loss: $(res_lbfgs.objective)")
                    end
                catch
                    println("LBFGS refinement failed, using ADAM results")
                end
            end

            println("Final training completed with:")
            println("  ADAM iterations: $adam_iters_final")
            println("  LBFGS iterations: $lbfgs_iters_final")
            println("  Learning rate: $(best_config[:adam_lr])")
            println("  Final loss: $(res_final.objective)")

            
            # Test on extended time
            t_test = range(0.0, 6π, length=200)
            
            # Analytical solution
            X_analytical_test = Array(solve(
                ODEProblem(harmonic_oscillator!, u0, (0.0, 6π), [ω_true]), 
                Tsit5(), 
                saveat=t_test, 
                abstol=1e-12, 
                reltol=1e-12
            ))

            # UDE prediction using BEST configuration found by Hyperband
            prob_test = remake(prob_nn_final, tspan=(0.0, 6π), p=res_final.u)
            sol_ude = solve(prob_test, best_config[:solver], saveat=t_test,
                        abstol=1e-8, reltol=1e-8)

            if sol_ude.retcode == :Success
                X_ude = Array(sol_ude)
                
                # Compute errors
                total_error = mean(abs2, X_ude .- X_analytical_test)
                println("Mean Squared Error vs Analytical: $(Printf.@sprintf("%.6f", total_error))")
                println("Using OPTIMAL configuration found by Hyperband:")
                println("  Solver: $(string(best_config[:solver]))")
                println("  Hidden dimensions: $(best_config[:hidden_dim])")
                println("  Layers: $(best_config[:n_layers])")
                println("  Activation: $(string(best_config[:activation]))")
                println("  Learning rate: $(best_config[:adam_lr])")
                println("  ADAM/LBFGS split: $(best_config[:adam_lbfgs_split])")
                println("  LBFGS max iterations: $(best_config[:lbfgs_max_iters])")
                
                # Plot results with enhanced information
                p1 = plot(t_test, X_analytical_test[1, :], label="Analytical Solution", 
                        color=:black, linewidth=2)
                plot!(p1, t_test, X_ude[1, :], 
                    label="UDE ($(string(best_config[:solver])))", 
                    color=:blue, linestyle=:dash, linewidth=2)
                scatter!(p1, t_train, X_noisy[1, :], 
                        label="Training data ($(100*noise_level)% noise)", 
                        color=:red, alpha=0.5, markersize=2)
                xlabel!(p1, "Time")
                ylabel!(p1, "Position x(t)")
                title!(p1, "Position: UDE vs Analytical (ω = $ω_true)")
                
                p2 = plot(t_test, X_analytical_test[2, :], label="Analytical Solution", 
                        color=:black, linewidth=2)
                plot!(p2, t_test, X_ude[2, :], 
                    label="UDE ($(string(best_config[:solver])))", 
                    color=:blue, linestyle=:dash, linewidth=2)
                scatter!(p2, t_train, X_noisy[2, :], 
                        label="Training data ($(100*noise_level)% noise)", 
                        color=:red, alpha=0.5, markersize=2)
                xlabel!(p2, "Time")
                ylabel!(p2, "Velocity v(t)")
                title!(p2, "Velocity: UDE vs Analytical")
                
                # Add error information to the plot
                plot_title = "Harmonic Oscillator UDE - MSE: $(Printf.@sprintf("%.2e", total_error))"
                final_plot = plot(p1, p2, layout=(1, 2), size=(1200, 500), 
                                plot_title=plot_title)
                
                # Add text annotation with key hyperparameters
                annotate!(p1, 0.8*6π, 0.8*maximum(X_analytical_test[1, :]), 
                        text("Hidden: $(best_config[:hidden_dim])\nLayers: $(best_config[:n_layers])", 
                            :right, 8, :gray))
                
                display(final_plot)
                
                # Optional: Save the plot
                savefig(final_plot, "harmonic_oscillator_ude_hyperband.png")
                
                # Additional analysis: Show performance in training region vs extrapolation
                train_indices = findall(x -> x <= maximum(t_train), t_test)
                extrap_indices = findall(x -> x > maximum(t_train), t_test)
                
                if !isempty(train_indices) && !isempty(extrap_indices)
                    train_error = mean(abs2, X_ude[:, train_indices] .- X_analytical_test[:, train_indices])
                    extrap_error = mean(abs2, X_ude[:, extrap_indices] .- X_analytical_test[:, extrap_indices])
                    
                    println("\nDetailed Error Analysis:")
                    println("  Training region (t ≤ $(maximum(t_train))): MSE = $(Printf.@sprintf("%.6f", train_error))")
                    println("  Extrapolation region (t > $(maximum(t_train))): MSE = $(Printf.@sprintf("%.6f", extrap_error))")
                    println("  Extrapolation/Training error ratio: $(Printf.@sprintf("%.2f", extrap_error/train_error))")
                end
                
            else
                println("UDE prediction failed with solver: $(string(best_config[:solver]))")
                println("Solver return code: $(sol_ude.retcode)")
            end
        end
        
    else
        println("\nHyperband failed to find a valid configuration (all losses >= 1e8)")
        println("Total function evaluations: $eval_count")
    end

    println("\n" * "="^60)
    println("OPTIMIZATION COMPLETE")
    println("="^60)
