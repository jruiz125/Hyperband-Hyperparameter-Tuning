# ---------------------------------------------------------------------------
# Clamped Pinned Rod UDE with Modern Global Optimizers
# 
# Implements better alternatives to ADAM with fewer hyperparameters
# ---------------------------------------------------------------------------
# Initialization
    # Setup project environment and configuration
    include("../utils/project_utils.jl")

    # Setup project environment
    project_root = setup_project_environment(activate_env = true, instantiate = false)

    # Create output directories for plots and results
    plots_dir = joinpath(project_root, "src", "Data", "NODE_Oscar_1MLPx3_GlobalOpt", "Plots_Results")
    mkpath(plots_dir)
    println("Plots & Results will be saved to: ", plots_dir)

    # This script is designed to run on a single thread Julia environment
    println("Running with $(Threads.nthreads()) threads")
    @assert Threads.nthreads() == 1 "Julia must be started with one threads!"

# -----------------------------------------------------------------------------
# Load Packages
    using Lux
    using Optimisers
    using LaTeXStrings
    using Plots
    using BenchmarkTools
    gr()

    # SciML Tools
    using OrdinaryDiffEq, SciMLSensitivity
    using Optimization, OptimizationOptimisers, OptimizationOptimJL

    # Global optimization packages
    using Evolutionary  # For Differential Evolution, PSO, CMA-ES
    using BlackBoxOptim # Alternative black-box optimization
    using Flux         # For additional optimizers like AdaBelief

    # Standard Libraries
    using LinearAlgebra, Statistics
    using Random

    # External Libraries
    using ComponentArrays, Lux, Zygote, StableRNGs
    using CpuId

    # ---------------------------------------------------------------------------
    # Print System Information
    println("\n=== System Information ===")
    begin
        cpu_name = CpuId.cpubrand()
        println("CPU: ", cpu_name)
        println("Physical cores: ", CpuId.cpucores())
        println("Logical cores (threads): ", CpuId.cputhreads())
        println("Number of Julia threads: ", Threads.nthreads())
    end
    println("=== System Information ===\n")

# -----------------------------------------------------------------------------
# A.- GENERATING Learning Data from the Ground-Truth data in MATLAB file
    using MATLAB

    # Use relative path based on project root for repository portability
    filename_DataSet = joinpath(project_root, "dataset", "LearnigData_Rod_Clamp_Pin_Rot_X02_Y00_mode2_train_085.mat")
    println("Loading dataset from: ", filename_DataSet)

    # Verify file exists before attempting to load
    if !isfile(filename_DataSet)
        error("Dataset file not found at: $filename_DataSet")
    end

    mf_DataSet = MatFile(filename_DataSet)
    data_DataSet = get_mvariable(mf_DataSet, "DataSet_train")
    DataSet = jarray(data_DataSet)

    # Dynamically determine the number of trajectories from the dataset
    num_trajectories = size(DataSet, 1)
    println("Found $num_trajectories trajectories in the dataset")

    # Configuration: You can modify this to use fewer trajectories if needed
    max_trajectories_to_use = nothing  # Change to a number like 10 for testing
    #max_trajectories_to_use = 10  

    # Determine actual number of trajectories to use
    trajectories_to_use = if max_trajectories_to_use === nothing
        num_trajectories
    else
        min(max_trajectories_to_use, num_trajectories)
    end

    println("Using $trajectories_to_use out of $num_trajectories available trajectories")

    # Initialize variables
    θ₀_true, κ₀_true = [], []
    px_true, py_true, θ_true, κ_true = Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[]

    # Load θ₀ and κ₀ from data 
    for i in 1:trajectories_to_use
        θ₀_i  = Float32.(DataSet[i,3])
        κ₀_i  = Float32.(DataSet[i,112])
        push!(θ₀_true,  θ₀_i)
        push!(κ₀_true,  κ₀_i)
    end
    θ₀_true = Float32.(collect(θ₀_true))
    κ₀_true = Float32.(collect(κ₀_true))

    # Load pX(s), pY(s), θ(s) and κ(s) from data
    ii = trajectories_to_use
    for i in 1:ii
        pX_i = Float32.(DataSet[i,12:61])
        pY_i = Float32.(DataSet[i,62:111])
        θ_i  = Float32.(DataSet[i,162:211])
        κ_i  = Float32.(DataSet[i,112:161])
        push!(px_true, pX_i)
        push!(py_true, pY_i)
        push!(θ_true,  θ_i)
        push!(κ_true,  κ_i)
    end

    # Combine all trajectories into a single Learning dataset
    X_ssol_all = [hcat(px_true[i], py_true[i], θ_true[i], κ_true[i])' for i in 1:ii]

# ----------------------------------------------------------------------------
# B.- TRAINING of the UODE with Modern Global Optimizers

    # Set a random seed for reproducible behaviour
    rng = StableRNG(1111)

    # Definition of the Universal Ordinary Differential Equation
    # Choose number of inputs, outputs
    m = 4
    n = 3

    # Define hyperparameters for the MLP neural network
    layers = collect([m, 20, 20, 20, n])

    # Multilayer FeedForward
    const U = Lux.Chain(
        [Dense(fan_in => fan_out, Lux.tanh) for (fan_in, fan_out) in zip(layers[1:end-2], layers[2:end-1])]...,
        Dense(layers[end-1] => layers[end], identity),
    ) 

    # Get the initial parameters and state variables of the model
    p, st = Lux.setup(rng, U)
    p = ComponentArray{Float32}(p)
    const _st = st

    # Global counters for NN passes
    const NN_FORWARD_COUNT = Ref(0)
    const NN_BACKWARD_COUNT = Ref(0)

    # Define the ODE
    function ude_dynamics!(du, u, p, s)
        NN_FORWARD_COUNT[] += 1
        # Add numerical stability checks
        if any(!isfinite, u)
            du .= 0.0f0
            return nothing
        end
        # Network prediction
        û_1 = U(u, p, _st)[1] 
        
        # Evaluate differential equations
        du[1] = û_1[1]   # dx/ds
        du[2] = û_1[2]   # dy/ds
        du[3] = u[4]     # dθ/ds = κ
        du[4] = û_1[3]   # dκ/ds
        return nothing
    end 

    # Sampling & model parameter space
    sSpan = (0.0f0, 1.0f0)
    s = Float32.(range(0, 1, length=50))

    # Use the first trajectory's initial state to define prob_nn
    u0 = Float32.(X_ssol_all[1][:, 1])
    const prob_nn = ODEProblem(ude_dynamics!, u0, sSpan, p)

    # Prediction function
    function predict(θ, X_all = X_ssol_all, S = s)
        try
            [Array(solve(remake(prob_nn, u0 = Float32.(X[:, 1]), tspan = (S[1], S[end]), p = θ),    
                    AutoVern7(Rodas5()), saveat = S, abstol = 1e-6, reltol = 1e-6,                  
                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))) for X in X_all]
        catch e
            # Return large loss if ODE fails
            return nothing
        end
    end

    # Loss function
    function loss(θ)
        X̂_sols = predict(θ)
        if X̂_sols === nothing
            return 1e10  # Large penalty for failed ODE solve
        end
        total_loss = sum(mean(abs, X_ssol_all[i] .- X̂_sols[i]) for i in 1:length(X̂_sols))
        return total_loss / length(X̂_sols)
    end

    # Store results for comparison
    results = Dict{String, Any}()
    losses_dict = Dict{String, Vector{Float64}}()

    # ----------------------------------------------------------------------------
    # METHOD 1: RAdam (Rectified Adam) - Best ADAM variant with minimal tuning
        println("\n" * "="^60)
        println("METHOD 1: RAdam - Rectified Adam (Auto-adaptive)")
        println("="^60)
        println("Hyperparameters: learning_rate only (auto-adapts warmup)")

        # RAdam automatically handles the adaptive learning rate warmup
        losses_radam = Float64[]
        callback_radam = function (p, l)
            push!(losses_radam, l)
            if length(losses_radam) % 50 == 0
                println("RAdam - Iteration $(length(losses_radam)): Loss = $(losses_radam[end])")
            end
            return false
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x, q) -> loss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, ComponentArray{Float32}(p))

        radam_time = @elapsed begin
            # RAdam needs only learning rate - automatically handles warmup
            res_radam = Optimization.solve(optprob, 
                                        Optimisers.RAdam(0.001),  # Only 1 hyperparameter!
                                        callback = callback_radam, 
                                        maxiters = 500)
        end

        results["RAdam"] = (params = res_radam.u, loss = losses_radam[end], time = radam_time)
        losses_dict["RAdam"] = losses_radam
        println("RAdam completed in $(round(radam_time, digits=2)) seconds")

    # ----------------------------------------------------------------------------
    # METHOD 2: AdaBelief - Adapts to the "belief" in gradient direction
        println("\n" * "="^60)
        println("METHOD 2: AdaBelief - Adaptive Belief Optimizer")
        println("="^60)
        println("Hyperparameters: learning_rate, β₂ (β₁ usually fixed)")

        losses_adabelief = Float64[]
        callback_adabelief = function (p, l)
            push!(losses_adabelief, l)
            if length(losses_adabelief) % 50 == 0
                println("AdaBelief - Iteration $(length(losses_adabelief)): Loss = $(losses_adabelief[end])")
            end
            return false
        end

        # Reset initial parameters
        optprob_adabelief = Optimization.OptimizationProblem(optf, ComponentArray{Float32}(p))

        adabelief_time = @elapsed begin
            # AdaBelief - adapts step size based on "belief" in gradient
            res_adabelief = Optimization.solve(optprob_adabelief,
                                            Optimisers.AdaBelief(0.001, (0.9, 0.999)),  # 2 main hyperparameters
                                            callback = callback_adabelief,
                                            maxiters = 500)
        end

        results["AdaBelief"] = (params = res_adabelief.u, loss = losses_adabelief[end], time = adabelief_time)
        losses_dict["AdaBelief"] = losses_adabelief
        println("AdaBelief completed in $(round(adabelief_time, digits=2)) seconds")

    # ----------------------------------------------------------------------------
    # METHOD 3: RMSprop - Simpler than ADAM, often better
        println("\n" * "="^60)
        println("METHOD 3: RMSprop - Root Mean Square Propagation")
        println("="^60)
        println("Hyperparameters: learning_rate, decay_rate")

        losses_rmsprop = Float64[]
        callback_rmsprop = function (p, l)
            push!(losses_rmsprop, l)
            if length(losses_rmsprop) % 50 == 0
                println("RMSprop - Iteration $(length(losses_rmsprop)): Loss = $(losses_rmsprop[end])")
            end
            return false
        end

        optprob_rmsprop = Optimization.OptimizationProblem(optf, ComponentArray{Float32}(p))

        rmsprop_time = @elapsed begin
            # RMSprop - only 2 hyperparameters
            res_rmsprop = Optimization.solve(optprob_rmsprop,
                                            Optimisers.RMSProp(0.001, 0.9),  # lr, decay
                                            callback = callback_rmsprop,
                                            maxiters = 500)
        end

        results["RMSprop"] = (params = res_rmsprop.u, loss = losses_rmsprop[end], time = rmsprop_time)
        losses_dict["RMSprop"] = losses_rmsprop
        println("RMSprop completed in $(round(rmsprop_time, digits=2)) seconds")

    # ----------------------------------------------------------------------------
    # METHOD 4: Differential Evolution - Gradient-free global optimization
        println("\n" * "="^60)
        println("METHOD 4: Differential Evolution (Gradient-Free)")
        println("="^60)
        println("Hyperparameters: population_size, F (mutation), CR (crossover)")

        # Convert to optimization problem for Evolutionary.jl
        function loss_vec(θ_vec)
            θ_comp = ComponentArray{Float32}(θ_vec, axes(p))
            return loss(θ_comp)
        end

        # Bounds for parameters (important for DE)
        lower_bounds = fill(-5.0, length(vec(p)))
        upper_bounds = fill(5.0, length(vec(p)))

        de_losses = Float64[]
        de_time = @elapsed begin
            # Differential Evolution - robust global optimizer
            result_de = Evolutionary.optimize(
                loss_vec,
                BoxConstraints(lower_bounds, upper_bounds),
                DE(;
                    populationSize = 50,      # Population size (main hyperparameter)
                    F = 0.8,                  # Mutation factor
                    CR = 0.9                  # Crossover probability
                ),
                Evolutionary.Options(
                    iterations = 100,         # Fewer iterations needed due to population
                    show_trace = true,
                    show_every = 10,
                    store_trace = true
                )
            )
            
            # Extract losses from trace
            for state in result_de.trace
                push!(de_losses, state.value)
            end
        end

        p_de = ComponentArray{Float32}(Evolutionary.minimizer(result_de), axes(p))
        results["DE"] = (params = p_de, loss = Evolutionary.minimum(result_de), time = de_time)
        losses_dict["DE"] = de_losses
        println("Differential Evolution completed in $(round(de_time, digits=2)) seconds")

    # ----------------------------------------------------------------------------
    # METHOD 5: Particle Swarm Optimization - Swarm intelligence
        println("\n" * "="^60)
        println("METHOD 5: Particle Swarm Optimization (PSO)")
        println("="^60)
        println("Hyperparameters: n_particles, w (inertia), c1, c2 (cognitive/social)")

        pso_losses = Float64[]
        pso_time = @elapsed begin
            # PSO - swarm-based global optimization
            result_pso = Evolutionary.optimize(
                loss_vec,
                BoxConstraints(lower_bounds, upper_bounds),
                PSO(;
                    n_particles = 30,         # Number of particles
                    w = 0.7,                  # Inertia weight
                    c1 = 1.5,                 # Cognitive parameter
                    c2 = 1.5                  # Social parameter
                ),
                Evolutionary.Options(
                    iterations = 100,
                    show_trace = true,
                    show_every = 10,
                    store_trace = true
                )
            )
            
            # Extract losses
            for state in result_pso.trace
                push!(pso_losses, state.value)
            end
        end

        p_pso = ComponentArray{Float32}(Evolutionary.minimizer(result_pso), axes(p))
        results["PSO"] = (params = p_pso, loss = Evolutionary.minimum(result_pso), time = pso_time)
        losses_dict["PSO"] = pso_losses
        println("PSO completed in $(round(pso_time, digits=2)) seconds")

    # ----------------------------------------------------------------------------
    # METHOD 6: CMA-ES - Covariance Matrix Adaptation Evolution Strategy
        println("\n" * "="^60)
        println("METHOD 6: CMA-ES (Self-Adaptive)")
        println("="^60)
        println("Hyperparameters: μ, λ (auto-computed from dimension)")

        cmaes_losses = Float64[]
        cmaes_time = @elapsed begin
            # CMA-ES - self-adaptive, minimal hyperparameters
            n_params = length(vec(p))
            μ = div(n_params, 4)  # Automatically set based on dimension
            λ = 2 * μ              # Automatically set
            
            result_cmaes = Evolutionary.optimize(
                loss_vec,
                BoxConstraints(lower_bounds, upper_bounds),
                CMAES(μ = μ, λ = λ),     # Auto-adaptive parameters!
                Evolutionary.Options(
                    iterations = 100,
                    show_trace = true,
                    show_every = 10,
                    store_trace = true
                )
            )
            
            # Extract losses
            for state in result_cmaes.trace
                push!(cmaes_losses, state.value)
            end
        end

        p_cmaes = ComponentArray{Float32}(Evolutionary.minimizer(result_cmaes), axes(p))
        results["CMAES"] = (params = p_cmaes, loss = Evolutionary.minimum(result_cmaes), time = cmaes_time)
        losses_dict["CMAES"] = cmaes_losses
        println("CMA-ES completed in $(round(cmaes_time, digits=2)) seconds")

    # ----------------------------------------------------------------------------
    # COMPARISON AND VISUALIZATION

        println("\n" * "="^60)
        println("GLOBAL OPTIMIZATION METHODS COMPARISON")
        println("="^60)
        println("Method\t\t| Final Loss\t| Time (s)\t| # Hyperparams")
        println("-"^60)
        for (name, result) in results
            println("$name\t| $(round(result.loss, digits=6))\t| $(round(result.time, digits=2))\t\t| $(
                name == "RAdam" ? 1 :
                name == "RMSprop" ? 2 :
                name == "AdaBelief" ? 2 :
                name == "DE" ? 3 :
                name == "PSO" ? 4 :
                name == "CMAES" ? 2 : "N/A"
            )")
        end
        println("="^60)

        # Plot comparison
        pl_comparison = Plots.plot(title="Global Optimizers Comparison", 
                                xlabel="Iterations", 
                                ylabel="Loss",
                                yaxis=:log10,
                                legend=:topright)

        colors = [:blue, :red, :green, :orange, :purple, :brown]
        for (idx, (name, losses)) in enumerate(losses_dict)
            if length(losses) > 0
                Plots.plot!(1:length(losses), losses, 
                        label=name, 
                        linewidth=2,
                        color=colors[idx])
            end
        end

        Plots.savefig(pl_comparison, joinpath(plots_dir, "global_optimizers_comparison.pdf"))
        display(pl_comparison)

        # Select best method
        best_method = argmin(Dict(name => result.loss for (name, result) in results))
        println("\n✅ Best method: $best_method with loss = $(results[best_method].loss)")
        p_trained_global = results[best_method].params

    # ----------------------------------------------------------------------------
    # HYBRID APPROACH: Best Global + Local Refinement

        println("\n" * "="^60)
        println("HYBRID: $(best_method) + LBFGS Refinement")
        println("="^60)

        # Use best global result as starting point for local refinement
        optprob_refine = Optimization.OptimizationProblem(optf, p_trained_global)

        refine_time = @elapsed begin
            res_refined = Optimization.solve(optprob_refine,
                                            LBFGS(linesearch = BackTracking()),
                                            maxiters = 1000)
        end

        println("Final refined loss: $(loss(res_refined.u))")
        println("Refinement time: $(round(refine_time, digits=2)) seconds")
        println("Total time: $(round(results[best_method].time + refine_time, digits=2)) seconds")

        p_trained = res_refined.u

    # ...existing code... (rest of the prediction and visualization code remains the same)