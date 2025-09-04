using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux, ComponentArrays, Zygote
using Plots
using Statistics, Random
using DataFrames
using BenchmarkTools
using JLD2
using Dates

# Include the Hyperband implementation
include("hypersolver.jl")

# -----------------------------------------------------------------------------
# Setup the Missing Physics Problem (Harmonic Oscillator with unknown term)
# -----------------------------------------------------------------------------

# True system parameters
const Ω = 0.1f0
function harmonic!(du, u, p, t)
    ω, ζ = p  # ω: natural frequency, ζ: damping ratio
    x, v = u  # position and velocity
    du[1] = dx = v
    du[2] = dv = -2ζ*ω*v - ω^2*x
end

# Initial condition
const u0 = [1.0f0, 0.0f0]  # Initial displacement and velocity

# Generate training data
const p_ = Float32[2.0, 0.1]  # Natural frequency = 2.0, damping ratio = 0.1
const tspan = (0.0f0, 3.0f0)
const t = range(tspan[1], tspan[2], length=40)
prob = ODEProblem(harmonic!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=t)

# Add noise to the data
X = Array(solution)
const Xₙ = X + Float32(1e-3) * randn(eltype(X), size(X))

# Plot the data
plot(solution, alpha=0.75, color=:black, label=["True Data" nothing])
scatter!(t, Xₙ', color=:red, label=["Noisy Data" nothing])

# -----------------------------------------------------------------------------
# Define the UDE Model
# -----------------------------------------------------------------------------

"""
Create a UDE model with given hyperparameters
"""
function create_ude_model(config)
    # Extract hyperparameters
    hidden_dim = config[:hidden_dim]
    n_layers = config[:n_layers]
    activation = config[:activation]
    
    # Build the neural network layers
    layers = []
    push!(layers, Dense(3, hidden_dim, activation))  # Input: [x, v, t]
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim, hidden_dim, activation))
    end
    push!(layers, Dense(hidden_dim, 1))  # Output: missing term
    
    # Create the model
    nn_model = Lux.Chain(layers...)
    
    rng = Random.default_rng()
    Random.seed!(rng, 1111)
    
    p_model, st = Lux.setup(rng, nn_model)
    
    # Define the UDE dynamics
    function ude_dynamics!(du, u, p, t, p_true)
        û = nn_model([u[1], u[2], t], p, st)[1]  # Network prediction
        du[1] = u[2]  # dx/dt = v (known physics)
        du[2] = û[1]  # dv/dt = neural network (missing physics)
    end
    
    # Closure to create the full dynamics
    nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
    
    # Create ODE problem
    prob_nn = ODEProblem(nn_dynamics!, u0, tspan, p_model)
    
    return prob_nn, nn_model, p_model, st
end

"""
Train UDE model with given configuration and resource budget
"""
function train_ude_model(config, resource)
    # Create model
    prob_nn, nn_model, p_init, st = create_ude_model(config)
    
    # Extract training hyperparameters
    lr = config[:learning_rate]
    solver = config[:solver]
    
    # Scale iterations based on resource
    max_iters = round(Int, resource)
    
    # Prediction function
    function predict(θ)
        _prob = remake(prob_nn, p=θ)
        Array(solve(_prob, solver, saveat=t,
                    abstol=1e-6, reltol=1e-6,
                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end
    
    # Loss function
    function loss(θ)
        X̂ = predict(θ)
        mean(abs2, Xₙ - X̂)
    end
    
    # Callback for early termination
    losses = Float32[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 10 == 0
            println("    Iteration $(length(losses)): Loss = $l")
        end
        return false
    end
    
    # Setup optimization
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
    
    # Train
    try
        res = Optimization.solve(optprob, ADAM(lr), callback=callback, maxiters=max_iters)
        final_loss = losses[end]
        return final_loss
    catch e
        println("    Training failed: $e")
        return Inf
    end
end

# -----------------------------------------------------------------------------
# Hyperparameter Configuration Space
# -----------------------------------------------------------------------------

function get_ude_config()
    Dict(
        :hidden_dim => rand([16, 32, 64, 128]),
        :n_layers => rand(2:5),
        :activation => rand([tanh, relu, sigmoid]),
        :learning_rate => 10.0^(-rand() * 3 - 1),  # 10^-4 to 10^-1
        :solver => rand([Tsit5(), Vern7(), AutoTsit5(Rosenbrock23())])
    )
end

# -----------------------------------------------------------------------------
# Run Hyperband vs Random Search Comparison
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("HYPERBAND vs RANDOM SEARCH for Missing Physics Discovery")
println("="^60)

# Total budget for fair comparison
const TOTAL_BUDGET = 1000  # Total training iterations available
const MAX_RESOURCE = 200   # Max iterations per configuration

"""
Run complete comparison experiment
"""
function run_comparison_experiment(n_runs=1)
    results = Dict(
        "Hyperband" => [],
        "Random Search" => []
    )
    
    for run in 1:n_runs
        println("\n" * "="^40)
        println("RUN $run/$n_runs")
        println("="^40)
        
        # 1. HYPERBAND
            println("\n--- Running Hyperband ---")
            eval_count = Ref(0)
            total_resource_used = Ref(0.0)
            
            function counted_ude_objective(config, resource)
                eval_count[] += 1
                total_resource_used[] += resource
                return train_ude_model(config, resource)
            end
            
            t_start = time()
            hb_config, hb_loss = hyperband(
                counted_ude_objective,
                get_ude_config,
                MAX_RESOURCE;
                η=3
            )
            hb_time = time() - t_start
            
            push!(results["Hyperband"], (
                loss=hb_loss,
                time=hb_time,
                evals=eval_count[],
                resource_used=total_resource_used[],
                config=hb_config
            ))
            
            println("Hyperband completed:")
            println("  Best loss: $hb_loss")
            println("  Time: $(round(hb_time, digits=2))s")
            println("  Evaluations: $(eval_count[])")
            println("  Total resource: $(total_resource_used[])")
        
        # 2. RANDOM SEARCH
            println("\n--- Running Random Search ---")
            eval_count[] = 0
            total_resource_used[] = 0.0
            
            t_start = time()
            rs_best_loss = Inf
            rs_best_config = nothing
            
            # Use same total budget as Hyperband
            n_configs = div(TOTAL_BUDGET, MAX_RESOURCE)
            
            for i in 1:n_configs
                config = get_ude_config()
                loss = counted_ude_objective(config, Float64(MAX_RESOURCE))
                if loss < rs_best_loss
                    rs_best_loss = loss
                    rs_best_config = config
                    println("  New best at config $i: loss = $rs_best_loss")
                end
            end
            
            rs_time = time() - t_start
            
            push!(results["Random Search"], (
                loss=rs_best_loss,
                time=rs_time,
                evals=eval_count[],
                resource_used=total_resource_used[],
                config=rs_best_config
            ))
            
            println("Random Search completed:")
            println("  Best loss: $rs_best_loss")
            println("  Time: $(round(rs_time, digits=2))s")
            println("  Evaluations: $(eval_count[])")
            println("  Total resource: $(total_resource_used[])")
    end
    
    return results
end

# Run the experiment
results = run_comparison_experiment(1)  # Reduced runs for speed

# -----------------------------------------------------------------------------
# Analysis and Visualization
# -----------------------------------------------------------------------------

    # Calculate statistics
    stats = DataFrame(
        Method = String[],
        Mean_Loss = Float64[],
        Std_Loss = Float64[],
        Min_Loss = Float64[],
        Mean_Time = Float64[],
        Std_Time = Float64[],
        Mean_Evals = Float64[],
        Mean_Resource = Float64[]
    )

    for method in keys(results)
        losses = [r.loss for r in results[method]]
        times = [r.time for r in results[method]]
        evals = [r.evals for r in results[method]]
        resources = [r.resource_used for r in results[method]]
        
        push!(stats, (
            method,
            mean(losses),
            std(losses),
            minimum(losses),
            mean(times),
            std(times),
            mean(evals),
            mean(resources)
        ))
    end

    println("\n" * "="^60)
    println("STATISTICAL SUMMARY")
    println("="^60)
    display(stats)

    # Create comparison plots
    p1 = bar(
        stats.Method,
        stats.Mean_Loss,
        yerr=stats.Std_Loss,
        title="Average Loss",
        ylabel="MSE Loss",
        legend=false,
        color=[:blue, :red]
    )

    p2 = bar(
        stats.Method,
        stats.Mean_Time,
        yerr=stats.Std_Time,
        title="Average Runtime",
        ylabel="Time (seconds)",
        legend=false,
        color=[:blue, :red]
    )

    p3 = bar(
        stats.Method,
        stats.Mean_Evals,
        title="Average Configurations Tested",
        ylabel="# Configurations",
        legend=false,
        color=[:blue, :red]
    )

    p4 = bar(
        stats.Method,
        stats.Mean_Resource,
        title="Average Total Resource Used",
        ylabel="Total Iterations",
        legend=false,
        color=[:blue, :red]
    )

    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
    display(combined_plot)
    savefig(combined_plot, "hyperband_vs_random_harmonic.png")

# -----------------------------------------------------------------------------
# Speedup Analysis
# -----------------------------------------------------------------------------

    speedup = stats[stats.Method .== "Random Search", :Mean_Time][1] / 
            stats[stats.Method .== "Hyperband", :Mean_Time][1]

    improvement = (stats[stats.Method .== "Random Search", :Mean_Loss][1] - 
                stats[stats.Method .== "Hyperband", :Mean_Loss][1]) / 
                stats[stats.Method .== "Random Search", :Mean_Loss][1] * 100

    println("\n" * "="^60)
    println("PERFORMANCE COMPARISON")
    println("="^60)
    println("Hyperband vs Random Search:")
    println("  Time Speedup: $(round(speedup, digits=2))x faster")
    println("  Loss Improvement: $(round(improvement, digits=1))%")
    println("  Resource Efficiency: Hyperband uses adaptive allocation")

# -----------------------------------------------------------------------------
# Visualize Best Found Model
# -----------------------------------------------------------------------------

    # Get best configuration from Hyperband
    best_run_idx = argmin([r.loss for r in results["Hyperband"]])
    best_config = results["Hyperband"][best_run_idx].config

    println("\n" * "="^60)
    println("BEST HYPERBAND CONFIGURATION")
    println("="^60)
    for (k, v) in best_config
        if k != :solver
            println("  $k: $v")
        else
            println("  $k: $(typeof(v))")
        end
    end

    # Train final model with best config and more resources
    println("\nTraining final model with best configuration...")
    prob_final, nn_final, p_final, st_final = create_ude_model(best_config)

    function predict_final(θ)
        _prob = remake(prob_final, p=θ)
        Array(solve(_prob, best_config[:solver], saveat=t,
                    abstol=1e-6, reltol=1e-6))
    end

    # Train for longer with best config
    final_loss = train_ude_model(best_config, 500.0)
    println("Final model loss after extended training: $final_loss")

    println("\nExperiment completed successfully!")
    println("Results saved to: hyperband_vs_random_harmonic.png")

# -------------- Plots -------------
    # Extract and display best configurations
    println("\n" * "="^60)
    println("OPTIMAL HYPERPARAMETERS FOUND")
    println("="^60)

    # Hyperband's best configuration
    hb_best_idx = argmin([r.loss for r in results["Hyperband"]])
    hb_best = results["Hyperband"][hb_best_idx]

    println("\n🏆 HYPERBAND BEST CONFIGURATION:")
    println("   Final Loss: $(hb_best.loss)")
    println("   Hyperparameters:")
    for (key, value) in hb_best.config
        if key == :activation
            println("   - $key: $(nameof(value))")
        elseif key == :solver
            println("   - $key: $(typeof(value))")
        else
            println("   - $key: $value")
        end
    end

    # Random Search's best configuration  
    rs_best_idx = argmin([r.loss for r in results["Random Search"]])
    rs_best = results["Random Search"][rs_best_idx]

    println("\n🎲 RANDOM SEARCH BEST CONFIGURATION:")
    println("   Final Loss: $(rs_best.loss)")
    println("   Hyperparameters:")
    for (key, value) in rs_best.config
        if key == :activation
            println("   - $key: $(nameof(value))")
        elseif key == :solver
            println("   - $key: $(typeof(value))")
        else
            println("   - $key: $value")
        end
    end

    # Compare the configurations
    println("\n" * "="^60)
    println("CONFIGURATION COMPARISON")
    println("="^60)

    comparison = DataFrame(
        Parameter = String[],
        Hyperband = Any[],
        Random_Search = Any[]
    )

    for key in keys(hb_best.config)
        hb_val = hb_best.config[key]
        rs_val = rs_best.config[key]
        
        if key == :activation
            push!(comparison, (string(key), nameof(hb_val), nameof(rs_val)))
        elseif key == :solver
            push!(comparison, (string(key), string(typeof(hb_val)), string(typeof(rs_val))))
        else
            push!(comparison, (string(key), hb_val, rs_val))
        end
    end

    display(comparison)
#=

    # -----------------------------------------------------------------------------
    # Save Best Hyperband Configuration using JLD2
    # -----------------------------------------------------------------------------

        println("\n" * "="^60)
        println("SAVING BEST HYPERBAND CONFIGURATION")
        println("="^60)

        # Get the best Hyperband model components
        best_config = hb_best.config
        
        # Recreate the best model to get final trained parameters
        prob_best, nn_best, p_best_init, st_best = create_ude_model(best_config)
        
        # Train the final model with extended iterations for optimal performance
        println("Training final best model...")
        final_loss_best = train_ude_model(best_config, 500.0)
        
        # Save the complete model state
        model_save_data = Dict(
            # Model architecture and configuration
            "config" => best_config,
            "neural_network" => nn_best,
            "initial_parameters" => p_best_init,
            "model_state" => st_best,
            
            # Training results
            "best_loss" => hb_best.loss,
            "final_extended_loss" => final_loss_best,
            
            # Performance metrics
            "hyperband_time" => hb_best.time,
            "hyperband_evaluations" => eval_count[],
            "hyperband_resource_used" => total_resource_used[],
            
            # Problem setup
            "problem_setup" => Dict(
                "u0" => u0,
                "tspan" => tspan,
                "true_parameters" => p_,
                "training_times" => collect(t),
                "noisy_data" => Xₙ
            ),
            
            # Comparison results
            "comparison_stats" => stats,
            "random_search_best" => rs_best,
            
            # Metadata
            "timestamp" => string(now()),
            "julia_version" => string(VERSION),
            "description" => "Best Hyperband configuration for Harmonic Oscillator UDE"
        )
        
        # Save to JLD2 file
        save_filename = "hyperband_harmonic_oscillator_best_model.jld2"
        @save save_filename model_save_data
        
        println("Model saved to: $save_filename")
        println("Saved components:")
        println("  ✓ Neural network architecture and configuration")
        println("  ✓ Trained model parameters and states")
        println("  ✓ Training results and performance metrics")
        println("  ✓ Problem setup and training data")
        println("  ✓ Comparison statistics")
        
        # Verification: Load and display basic info
        loaded_data = load(save_filename)["model_save_data"]
        println("\nVerification - Loaded model info:")
        println("  Best loss: $(loaded_data["best_loss"])")
        println("  Configuration: $(loaded_data["config"])")
        println("  Timestamp: $(loaded_data["timestamp"])")

    # -----------------------------------------------------------------------------
    # Comprehensive Analysis: NN vs Analytical Solution
    # -----------------------------------------------------------------------------

    println("\n" * "="^60)
    println("COMPREHENSIVE ANALYSIS: NN vs ANALYTICAL SOLUTION")
    println("="^60)

    # Define analytical solution for damped harmonic oscillator
    function analytical_solution(t, u0, p)
        ω, ζ = p  # natural frequency, damping ratio
        x0, v0 = u0
        
        if ζ < 1.0  # Underdamped case
            ωd = ω * sqrt(1 - ζ^2)  # damped frequency
            A = x0
            B = (v0 + ζ*ω*x0) / ωd
            
            x_analytical = exp.(-ζ*ω*t) .* (A*cos.(ωd*t) + B*sin.(ωd*t))
            v_analytical = exp.(-ζ*ω*t) .* ((-ζ*ω*A - ωd*B)*cos.(ωd*t) + (-ζ*ω*B + ωd*A)*sin.(ωd*t))
        elseif ζ == 1.0  # Critically damped
            A = x0
            B = v0 + ω*x0
            
            x_analytical = exp.(-ω*t) .* (A .+ B*t)
            v_analytical = exp.(-ω*t) .* (B - ω*(A .+ B*t))
        else  # Overdamped
            r1 = -ω*(ζ + sqrt(ζ^2 - 1))
            r2 = -ω*(ζ - sqrt(ζ^2 - 1))
            A = (v0 - r2*x0) / (r1 - r2)
            B = x0 - A
            
            x_analytical = A*exp.(r1*t) + B*exp.(r2*t)
            v_analytical = A*r1*exp.(r1*t) + B*r2*exp.(r2*t)
        end
        
        return [x_analytical'; v_analytical']
    end

    # Create extended time series for comprehensive analysis
    t_extended = range(0.0f0, 8.0f0, length=200)
    t_train = collect(t)
    t_extrap = t_extended[t_extended .> maximum(t)]

    # Get analytical solutions
    X_analytical_train = analytical_solution(t_train, u0, p_)
    X_analytical_extended = analytical_solution(t_extended, u0, p_)

    # Get best model predictions
    best_config = hb_best.config
    prob_best, nn_best, p_final, st_best = create_ude_model(best_config)

    # Train final model to get best parameters
    final_loss = train_ude_model(best_config, 500.0)

    # Create prediction function with trained model
    function predict_best_model(θ, time_span, time_points)
        _prob = remake(prob_best, p=θ, tspan=(time_span[1], time_span[end]))
        sol = solve(_prob, best_config[:solver], saveat=time_points, 
                    abstol=1e-8, reltol=1e-6)
        return Array(sol)
    end

    # Get predictions for different time ranges
    X_pred_train = predict_best_model(p_final, (t_train[1], t_train[end]), t_train)
    X_pred_extended = predict_best_model(p_final, (t_extended[1], t_extended[end]), t_extended)

    # Calculate L2-norm errors
    l2_error_train = sqrt.(mean((X_pred_train - X_analytical_train).^2, dims=2))
    l2_error_extended = sqrt.(mean((X_pred_extended - X_analytical_extended).^2, dims=2))

    println("L2-norm errors:")
    println("  Position (training): $(l2_error_train[1])")
    println("  Velocity (training): $(l2_error_train[2])")
    println("  Position (extended): $(l2_error_extended[1])")
    println("  Velocity (extended): $(l2_error_extended[2])")

    # Calculate point-wise errors for plotting
    error_position = abs.(X_pred_extended[1,:] - X_analytical_extended[1,:])
    error_velocity = abs.(X_pred_extended[2,:] - X_analytical_extended[2,:])

    # Create comprehensive comparison plots
    println("\nGenerating comprehensive comparison plots...")

    # Plot 1: Population Dynamics Comparison
    p1 = plot(layout=(1,2), size=(1200, 400), margin=5Plots.mm)

    plot!(p1[1], t_extended, X_analytical_extended[1,:], 
        label="True Position", color=:black, linewidth=2)
    plot!(p1[1], t_extended, X_pred_extended[1,:], 
        label="Hyperband Prediction", color=:blue, linewidth=2, linestyle=:dash)
    plot!(p1[1], t_train, Xₙ[1,:], 
        seriestype=:scatter, label="Training Data", color=:red, markersize=3)
    vline!(p1[1], [maximum(t_train)], color=:orange, linestyle=:dash, linewidth=2, 
        label="Training/Extrapolation")
    title!(p1[1], "Position Dynamics Comparison")
    xlabel!(p1[1], "Time")
    ylabel!(p1[1], "Position")

    plot!(p1[2], t_extended, X_analytical_extended[2,:], 
        label="True Velocity", color=:black, linewidth=2)
    plot!(p1[2], t_extended, X_pred_extended[2,:], 
        label="Hyperband Prediction", color=:blue, linewidth=2, linestyle=:dash)
    plot!(p1[2], t_train, Xₙ[2,:], 
        seriestype=:scatter, label="Training Data", color=:red, markersize=3)
    vline!(p1[2], [maximum(t_train)], color=:orange, linestyle=:dash, linewidth=2, 
        label="Training/Extrapolation")
    title!(p1[2], "Velocity Dynamics Comparison")
    xlabel!(p1[2], "Time")
    ylabel!(p1[2], "Velocity")

    # Plot 2: Phase Portrait Comparison
    p2 = plot(size=(600, 600), margin=5Plots.mm)
    plot!(p2, X_analytical_extended[1,:], X_analytical_extended[2,:], 
        label="Ground Truth", color=:black, linewidth=3)
    plot!(p2, X_pred_extended[1,:], X_pred_extended[2,:], 
        label="Hyperband", color=:blue, linewidth=2, linestyle=:dash)
    scatter!(p2, [u0[1]], [u0[2]], label="Initial Condition", 
            color=:green, markersize=8, markershape=:star)
    title!(p2, "Phase Portrait Comparison")
    xlabel!(p2, "Position")
    ylabel!(p2, "Velocity")

    # Plot 3: Training Convergence (reuse existing data)
    p3 = plot(size=(600, 400), margin=5Plots.mm)
    if length(losses_baseline) > 0
        plot!(p3, 1:length(losses_baseline), losses_baseline, 
            label="Baseline ADAM", color=:green, linewidth=2, yscale=:log10)
    end
    # Note: We don't have convergence data for the best model training stored
    # This would need to be captured during the training process
    title!(p3, "Training Convergence Comparison")
    xlabel!(p3, "Iterations")
    ylabel!(p3, "Loss")

    # Plot 4: Prediction Error Over Time
    p4 = plot(size=(600, 400), margin=5Plots.mm)
    plot!(p4, t_extended, error_position, 
        label="Position Error", color=:red, linewidth=2)
    plot!(p4, t_extended, error_velocity, 
        label="Velocity Error", color=:blue, linewidth=2)
    vline!(p4, [maximum(t_train)], color=:orange, linestyle=:dash, linewidth=2, 
        label="Training/Extrapolation")
    title!(p4, "Prediction Error Over Time")
    xlabel!(p4, "Time")
    ylabel!(p4, "L2 Error")
    yscale!(p4, :log10)

    # Plot 5: Missing Physics Recovery - Neural Network Output
    p5 = plot(layout=(1,2), size=(1200, 400), margin=5Plots.mm)

    # Calculate what the network is learning (the missing acceleration term)
    function nn_output(u, p, t)
        return nn_best([u[1], u[2], t], p, st_best)[1][1]
    end

    # Calculate true missing term and NN prediction
    nn_predictions = []
    true_missing = []
    for i in 1:length(t_extended)
        u_curr = [X_analytical_extended[1,i], X_analytical_extended[2,i]]
        nn_pred = nn_output(u_curr, p_final, t_extended[i])
        
        # True acceleration from harmonic oscillator
        ω, ζ = p_
        true_acc = -2ζ*ω*u_curr[2] - ω^2*u_curr[1]
        
        push!(nn_predictions, nn_pred)
        push!(true_missing, true_acc)
    end

    plot!(p5[1], t_extended, true_missing, 
        label="True Acceleration", color=:black, linewidth=2)
    plot!(p5[1], t_extended, nn_predictions, 
        label="NN Prediction", color=:blue, linewidth=2, linestyle=:dash)
    vline!(p5[1], [maximum(t_train)], color=:orange, linestyle=:dash, linewidth=2, 
        label="Training/Extrapolation")
    title!(p5[1], "Missing Physics Recovery (Acceleration)")
    xlabel!(p5[1], "Time")
    ylabel!(p5[1], "Acceleration")

    # Error in missing physics recovery
    missing_error = abs.(nn_predictions - true_missing)
    plot!(p5[2], t_extended, missing_error, 
        label="NN Recovery Error", color=:red, linewidth=2)
    vline!(p5[2], [maximum(t_train)], color=:orange, linestyle=:dash, linewidth=2, 
        label="Training/Extrapolation")
    title!(p5[2], "Missing Physics Recovery Error")
    xlabel!(p5[2], "Time")
    ylabel!(p5[2], "Error")
    yscale!(p5[2], :log10)

    # Combine all plots
    final_plot = plot(p1, p2, p3, p4, p5, 
                    layout=@layout([a{0.4h}; b{0.3h} c{0.3h}; d{0.3h} e{0.3h}]), 
                    size=(1200, 1000))

    display(final_plot)
    savefig(final_plot, "harmonic_oscillator_comprehensive_analysis.png")

    # Save detailed analysis results
    analysis_results = Dict(
        "l2_errors" => Dict(
            "position_training" => l2_error_train[1],
            "velocity_training" => l2_error_train[2],
            "position_extended" => l2_error_extended[1],
            "velocity_extended" => l2_error_extended[2]
        ),
        "time_series" => Dict(
            "t_extended" => t_extended,
            "analytical_solution" => X_analytical_extended,
            "nn_prediction" => X_pred_extended,
            "training_data" => Xₙ,
            "training_times" => t_train
        ),
        "missing_physics" => Dict(
            "true_acceleration" => true_missing,
            "nn_acceleration" => nn_predictions,
            "recovery_error" => missing_error
        )
    )

    @save "harmonic_oscillator_analysis_results.jld2" analysis_results

    println("Comprehensive analysis completed!")
    println("  ✓ L2-norm errors calculated and displayed")
    println("  ✓ Phase portrait comparison generated")
    println("  ✓ Prediction vs analytical solution plotted")
    println("  ✓ Missing physics recovery analyzed")
    println("  ✓ Results saved to: harmonic_oscillator_comprehensive_analysis.png")
    println("  ✓ Analysis data saved to: harmonic_oscillator_analysis_results.jld2")

    println("\n" * "="^60)
    println("HARMONIC OSCILLATOR HYPERBAND OPTIMIZATION COMPLETED")
    println("="^60)
    println("Best model and configuration saved successfully!")
    println("Results available in: hyperband_vs_random_harmonic.png")
    println("Model saved to: $save_filename")
    =#