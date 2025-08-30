# ---------------------------------------------------------------------------
# Clamped Pinned Rod UDE
# 
# Solves the inverse position problem of a Clamped-pinned Cosserat beam using Neural ODE Networks. UDE methodology is implemented.
# ---------------------------------------------------------------------------

#using Pkg
#Pkg.develop(path=joinpath(homedir(), ".julia", "dev", "ClampedPinnedRodSolver"))
#using ClampedPinnedRodSolver

# Setup project environment and configuration
include("../utils/project_utils.jl")

# Setup project environment
project_root = setup_project_environment(activate_env = true, instantiate = false)

# Create output directories for plots and results
plots_dir = joinpath(project_root, "src", "Data", "NODE_Oscar_1MLPx3", "Plots_Results")
mkpath(plots_dir)
println("Plots & Results will be saved to: ", plots_dir)

# This script is designed to run on a single thread Julia environment
println("Running with $(Threads.nthreads()) threads")
@assert Threads.nthreads() == 1 "Julia must be started with one threads!"


# -----------------------------------------------------------------------------
# 0 .- Load Packages
#using ClampedPinnedRodUDE
    using Lux #, LuxCUDA  # The deep learning (neural network) framework. I do not use GPU now.
    using Optimisers
    using LaTeXStrings
    using Plots
    using BenchmarkTools  # Added for timing the training process
    gr()
        # SciML Tools
            using OrdinaryDiffEq, SciMLSensitivity
            using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

        # Standard Libraries
            using LinearAlgebra, Statistics

        # External Libraries
            using ComponentArrays, Lux, Zygote, StableRNGs

        using CpuId
# ---------------------------------------------------------------------------
# Print System Information
    println("\n=== System Information ===")
    # 
    begin
        cpu_name = CpuId.cpubrand()
        println("CPU: ", cpu_name)
        println("Physical cores: ", CpuId.cpucores())
        println("Logical cores (threads): ", CpuId.cputhreads())
        println("Number of Julia threads: ", Threads.nthreads())
    end
    println("=== System Information ===")
        
# -----------------------------------------------------------------------------
# A.- GENERATING Learning Data from the Ground-Truth data in MATLAB file: (px(θ₀), py(θ₀), κ₀(θ₀), θ₀) 
    
    # Reading .mat file  
        using MATLAB

        # Use relative path based on project root for repository portability
        filename_DataSet = joinpath(project_root, "dataset", "LearnigData_Rod_Clamp_Pin_Rot_X05_Y00_mode2_train_085.mat")
        println("Loading dataset from: ", filename_DataSet)
        
        # Verify file exists before attempting to load
        if !isfile(filename_DataSet)
            error("Dataset file not found at: $filename_DataSet")
        end
        
        mf_DataSet = MatFile(filename_DataSet)                # opens a Matlab file for reading
        data_DataSet = get_mvariable(mf_DataSet, "DataSet_train")   # gets a variable and returns an mxArray
        DataSet = jarray(data_DataSet)                        # converts x to a Julia matrix
        
        # Dynamically determine the number of trajectories from the dataset
        num_trajectories = size(DataSet, 1)
        println("Found $num_trajectories trajectories in the dataset")
        
        # Configuration: You can modify this to use fewer trajectories if needed
        # Set to `nothing` to use all available trajectories, or specify a number
        max_trajectories_to_use = nothing  # Change to a number like 35 if you want to limit
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
        for i in 1:trajectories_to_use # Number of trajectories: dynamically determined
            θ₀_i  = Float32.(DataSet[i,3]);   # input: initial theta angle, convert to Float32.
            κ₀_i  = Float32.(DataSet[i,112]); # input: initial curvature, convert to Float32.
            # Append to lists
            push!(θ₀_true,  θ₀_i)
            push!(κ₀_true,  κ₀_i)
        end
        θ₀_true = Float32.(collect(θ₀_true))
        κ₀_true = Float32.(collect(κ₀_true))
        
    # Load pX(s), pY(s), θ(s) and κ(s) from data to create the Learning set. 
                # (s): 50 discrete values of s in DataSet
        ii = trajectories_to_use # Use the determined number of trajectories
        for i in 1:ii
            pX_i = Float32.(DataSet[i,12:61]  );   # input: position X, convert to Float32.
            pY_i = Float32.(DataSet[i,62:111] );   # input: position Y, convert to Float32.
            θ_i  = Float32.(DataSet[i,162:211]);   # input: theta angle, convert to Float32.
            κ_i  = Float32.(DataSet[i,112:161]);   # input: curvature, convert to Float32.
            # Append to lists
            push!(px_true, pX_i)
            push!(py_true, pY_i)
            push!(θ_true,  θ_i)
            push!(κ_true,  κ_i)
        end

    # Combine all ii trajectories into a single Learning dataset
        X_ssol_all = [hcat(px_true[i], py_true[i], θ_true[i], κ_true[i])' for i in 1:ii]
            
# ----------------------------------------------------------------------------
# B.- TRAINING of the UODE

    # Set a random seed for reproducible behaviour
        rng = StableRNG(1111);

    # Definition of the Universal Ordinary Differential Equation
        # Choose number of inputs, outputs
        m = 4
        n = 3
        # Define hyperparameters for the MLP neural network
        layers = collect([m, 20, 20, 20, n])
  
        # Multilayer FeedForward
        const U = Lux.Chain(
            [Dense(fan_in => fan_out, Lux.tanh) for (fan_in, fan_out) in zip(layers[1:end-2], layers[2:end-1])]...,
            Dense(layers[end-1] => layers[end], identity),) 
            
            # Get the initial parameters and state variables of the model
            p, st = Lux.setup(rng, U)# network U initialization using random numbers
            p = ComponentArray{Float32}(p)  # Changed from Float64 to Float32
            const _st = st
        # Global counters for NN passes
            const NN_FORWARD_COUNT = Ref(0)
            const NN_BACKWARD_COUNT = Ref(0)
        # Define the ODE
            function ude_dynamics!(du, u, p, s)
                NN_FORWARD_COUNT[] += 1  # Count forward pass
                # Add numerical stability checks
                if any(!isfinite, u)
                    # If any state is non-finite, set derivatives to zero to stop integration
                    du .= 0.0f0
                    return nothing
                end
                # Current state.
                # x, y, θ, κ = u
                # Network prediction (forward pass), used to modify the derivatives of the system´s state: du[1] and du[2].
                û_1 = U(u, p, _st)[1] 
 
                # Evaluate differential equations.
                du[1] = û_1[1]   # du[1] = cos(θ)                                                                   
                du[2] = û_1[2]   # du[2] = sin(θ)                                                                    
                du[3] = u[4]     # du[3] = κ                                                           
                du[4] = û_1[3]   # du[4] = R / (E * I) * sin(θ - Ψ)                                                                 
                return nothing
            end 

    # Sampling & model parameter space
        sSpan = (0.0f0, 1.0f0)
        s = Float32.(range(0, 1, length=50))  # Explicitly create 50 points

    # Use the first trajectory's initial state to define prob_nn
        u0 = Float32.(X_ssol_all[1][:, 1])
        const prob_nn = ODEProblem(ude_dynamics!, u0, sSpan, p)
    
    # Update the predict function to handle the initial conditions and parameters from the optimizer
    function predict(θ, X_all = X_ssol_all, S = s)
        # Enable experimental debug mode to understand the issue better
        if haskey(ENV, "LUX_DEBUG")
            Lux.Experimental.@debug_mode true
        end
        [Array(solve(remake(prob_nn, u0 = Float32.(X[:, 1]), tspan = (S[1], S[end]), p = θ),    
                AutoVern7(Rodas5()), saveat = S, abstol = 1e-6, reltol = 1e-6,                  
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))) for X in X_all]
            end
    
    # Update the loss function to compute the loss over all trajectories
    function loss(θ)
        X̂_sols = predict(θ)
        total_loss = sum(mean(abs, X_ssol_all[i] .- X̂_sols[i]) for i in 1:length(X̂_sols))
        return total_loss / length(X̂_sols)
    end

    losses = Float64[]

    # Callback
        callback = function (p, l)
            push!(losses, l)
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            return false
        end

    # Training (solving optimization problem)
     # Define the optimizer    

        adtype = Optimization.AutoZygote()                                                 
        optf = Optimization.OptimizationFunction((x, q) -> loss(x), adtype)               
        optprob = Optimization.OptimizationProblem(optf, ComponentArray{Float32}(p))  # Changed to Float32     
        
        # Benchmark the training process
        println("Starting training benchmark...")
        training_time = @elapsed begin
            # ADAM loop
            learning_rate = 0.03
            epoch_1 = 200 # ADAM
            println("Starting ADAM optimization ($(epoch_1) iterations)...")
            adam_time = @elapsed (res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(learning_rate), callback = callback, maxiters = epoch_1))
            
            # LBFGS loop
            epoch_2 = 1000 # LBFGS
            println("Starting LBFGS optimization ($(epoch_2) iterations)...")
            optprob2 = Optimization.OptimizationProblem(optf, res1.u)
            lbfgs_time = @elapsed (res2 = Optimization.solve(optprob2, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = epoch_2))
            
            # Print timing results
            println("\n" * "="^60)
            println("TRAINING BENCHMARK RESULTS")
            println("="^60)
            println("ADAM phase:  $(round(adam_time, digits=2)) seconds ($(epoch_1) iterations)")
            println("LBFGS phase: $(round(lbfgs_time, digits=2)) seconds ($(epoch_2) iterations)")
            println("Total time:  $(round(adam_time + lbfgs_time, digits=2)) seconds")
            println("="^60)
        end

    # Final Losses Plot
        begin
            println("Final training loss after $(length(losses)) iterations: $(losses[end])")
            pl_losses = Plots.plot(1:epoch_1, losses[1:epoch_1], yaxis = :log10, xaxis = :log10,
                                xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
            Plots.plot!(epoch_1 + 1:length(losses), losses[epoch_1 + 1:end], yaxis = :log10, xaxis = :log10,
                        xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)
            #Plots.savefig(pl_losses, "theta_0_sol_1_mode_2_X02_losses.pdf")
            display(pl_losses)
        end

    # Rename the best candidate
        p_trained = res2.u # es la red neuronal entrenada (sus parámetros: pesos y bias)

# ----------------------------------------------------------------------------
# C.- MAKING Prediction (UODE):
     # Generate a New Initial state vector {x₀, y₀, θ₀, κ(θ₀)}

     x₀ = [0 for i in 1:ii]
     y₀ = [0 for i in 1:ii]
     
     X_ssssol = [hcat(x₀[i], y₀[i], θ₀_true[i], κ₀_true[i])' for i in 1:ii]
    
     # Choose rod pose to test comparison
     c = 15

     # Convert the 4x1 Adjoint matrix to a Vector{Float32}
        new_initial_state = vec(X_ssssol[c])

        new_initial_state = [new_initial_state[1]; new_initial_state[2]; new_initial_state[3]; new_initial_state[4]]

    # Predict using the trained parameters
        new_prediction = Array(solve(remake(prob_nn, u0 = new_initial_state, p = p_trained), 
                                    AutoVern7(Rodas5()), saveat = s, abstol = 1e-6, reltol = 1e-6, 
                                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))

    # Visualizing the new UDE prediction
        # Assign new names
        px_predict = new_prediction[1,1:end]
        py_predict = new_prediction[2,1:end]
        θ_predict  = new_prediction[3,1:end]
        κ_predict  = new_prediction[4,1:end]

    # Visualizing the UODE prediction vs. Ground Truth
        # Phase plot θ(s) vs. κ(s) of the rod - UODE prediction vs. Ground Truth
        ts = first(s):mean(diff(s)):last(s)
            # θ(s)_true - UDE prediction -
                begin
                    Pp = Plots.plot(ts, vec(θ_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best)
                    Plots.scatter!(ts, θ_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s \,\, [m]", ylabel=L"θ \,\, [rad]")
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_180º_mode_2_X02_Theta-s.pdf"))
                    display(Pp)
                end
            # κ(s)_true - UDE prediction -
                
            begin
                    Pp = Plots.plot(s, vec(κ_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best)
                    Plots.scatter!(s, κ_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s \,\, [m]", ylabel=L"\dot{\theta}\,\,=\,\,k\,\, [-]")
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_180º_mode_2_X02_Kappa-s.pdf"))
                    display(Pp)
                end

            # Phase diagrams comparison
                begin
                    f44 = Plots.plot(vec(θ_true[c]), vec(κ_true[c]), color=[:red :gray], legend=:best , label="Curva original")
                    Plots.scatter!(θ_predict, κ_predict, color=[:black :gray], markersize=3, legend=:best , label="UODE",
                            xlabel=L"θ \,\, [rad]", ylabel=L"κ\,\, [-] ")
                    Plots.savefig(f44, joinpath(plots_dir, "theta_0_sol_180º_mode_2_X02_phase_C_3MLP.pdf"))
                    display(f44)
                end
                
            # Error analysis θ(s), κ(s)
            θ_traj = θ_predict
            κ_traj = κ_predict
            # Global L1 & L2 error
            l1_error_θ = sum(abs.(θ_traj .- θ_true[c]))
            l2_error_θ = sqrt(sum((θ_traj .- θ_true[c]).^2))
            l1_error_κ = sum(abs.(κ_traj .- κ_true[c]))
            l2_error_κ = sqrt(sum((κ_traj .- κ_true[c]).^2))
        # Error for each point
            Errors_θ = abs.(θ_traj .- θ_true[c])
            Errors_κ = abs.(κ_traj .- κ_true[c])

    # Plot errors
        begin
            f55 = Plots.plot(1:length(Errors_θ), Errors_θ, color=:red, label=["θ error" nothing], 
                        xlabel=L"Nodos", ylabel=L"Error", title="")
            Plots.plot!(1:length(Errors_κ), Errors_κ, color=:blue, label=["κ error" nothing], 
                        xlabel=L"Nodos", ylabel=L"Error", title="")
            Plots.savefig(f55, joinpath(plots_dir, "theta_0_sol_180º_mode_2_X02_theta-kappa_error.pdf"))
            display(f55)
        end

        # Shape of the rod - U=DE prediction vs. Ground Truth 
            # x(s)
                begin
                    Pp = Plots.plot(s, vec(px_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(s, px_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s\,\,[m]", ylabel=L"x\,\,[m]")
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_180º_mode_2_X02_X-s.pdf"))
                    display(Pp)
                end
            # y(s)
            begin
                    Pp = Plots.plot(s, vec(py_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(s, py_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s\,\,[m]", ylabel=L"y\,\,[m]")
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_180º_mode_2_X02_Y-s.pdf"))
                    display(Pp)
                end
            # (x, y)
                begin
                    f66 = Plots.plot(vec(px_true[c]), vec(py_true[c]), color=[:black :gray], label=["Curva original" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(px_predict, py_predict, color=[:blue :orange], label=["UODE" nothing],
                        xlabel=L"x\,\,[m]", ylabel=L"y\,\,[m]")
                    Plots.savefig(f66, joinpath(plots_dir, "theta_0_sol_180º_mode_2_X02_X-Y.pdf"))
                    display(f66)
                end
            # Error analysis x(s), y(s)
                x_traj = px_predict
                y_traj = py_predict

                # Global L1 & L2 error
                    l1_error_2 = sum(abs.(x_traj .- px_true[c]) .+ abs.(y_traj .- py_true[c]))
                    l2_error_2 = sqrt(sum((x_traj .- px_true[c]).^2 + (y_traj .- py_true[c]).^2))

                # L1 & L2 error for each point
                    l1_errors_2 = abs.(x_traj .- px_true[c]) .+ abs.(y_traj .- py_true[c])
                    l2_errors_2 = sqrt.((x_traj .- px_true[c]).^2 .+ (y_traj .- py_true[c]).^2)

            # Plot errors
                begin
                    #Plots.plot(1:length(l1_errors_2), l1_errors_2, color=:red, label=L"L1\,\,Error" , marker=:vline, 
                    #               xlabel=L"Nodos", ylabel=L"L1\,\,&\,\,L2\,\,\,\,Error", title="")
                    f77 = Plots.plot(1:length(l2_errors_2), l2_errors_2, color=:red, label=nothing, 
                                xlabel=L"Nodos", ylabel=L"L2\,\,\,\,Error", title="")
                    #Plots.savefig(f77, "theta_0_sol_1_mode_2_X02_X-Y_error.pdf")
                    display(f77)
                end

        # Final plots
            # θ(s), κ(s) 
                begin
                f44_55 = Plots.plot(f44, f55, layout = (1, 2), size=(1200, 500), left_margin=10Plots.mm, right_margin=10Plots.mm, top_margin=10Plots.mm, bottom_margin=10Plots.mm)
                #Plots.savefig(f44_55, "theta_0_sol_1_mode_2_X02_theta-kappa_error_T_3.4.pdf")
                display(f44_55)
                end
            # x(s), y(s)
                begin
                f66_77 = Plots.plot(f66, f77, layout = (1, 2), size=(1200, 500), left_margin=10Plots.mm, right_margin=10Plots.mm, top_margin=10Plots.mm, bottom_margin=10Plots.mm)
                #Plots.savefig(f66_77, "theta_0_sol_1_mode_2_X02_X-Y_error_3.4.pdf")
                display(f66_77)
                end
    
# -------------------------------------------------------
# D.- Save NNs

    using JLD2

    model = U
    parameters = p_trained
    states = _st

    # Save the NN data to a file using relative path for portability
    output_file = joinpath(plots_dir, "NN_NODE_1MLPx3_tanh_X05_R1.jld2")
    println("Saving trained model to: ", output_file)

    # Create directory if it doesn't exist
    mkpath(dirname(output_file))

    @save output_file model parameters states
    println("✓ Model saved successfully to: ","$plots_dir")
# -------------------------------------------------------
