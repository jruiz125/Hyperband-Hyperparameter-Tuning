# ---------------------------------------------------------------------------
"
    Neural ODE approximation to the Kinematics of a Clamped-pinned Cosserat beam.
    Solution of the Inverse Problem from a home position as the Clamping is rotated.
    14/04/2025
    UPV-EHU
    Oscar Altuzarra
"
# ---------------------------------------------------------------------------
# Inicialization
    cd(@__DIR__)  # Changes the current working directory to the directory containing the script, this case to xxx directory.
    pwd()         # Directorio de trabajo actual
    using Pkg         
    Pkg.activate(".")
# -----------------------------------------------------------------------------
# 0 .- Load Packages
# using JLD2  # JLD2 saves and loads Julia data structures in a format comprising a subset of HDF5
using Lux #, LuxCUDA  # The deep learning (neural network) framework. I do not use GPU now.
using Optimisers
using LaTeXStrings
using Plots
gr()
    # SciML Tools
        using OrdinaryDiffEq, SciMLSensitivity
        using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

    # Standard Libraries
        using LinearAlgebra, Statistics

    # External Libraries
        using ComponentArrays, Lux, Zygote, StableRNGs
        
# -----------------------------------------------------------------------------
# A.- GENERATING Learning Data from the Ground-Truth data in MATLAB file: (px(θ₀), py(θ₀), κ₀(θ₀), θ₀) 
    
    # Reading .mat file  
        using MATLAB
        ENV["MATLAB_ROOT"] = raw"C:\Program Files\MATLAB\R2023a\bin" # on my Windows machine
        filename_DataSet =  "C:/Users/impalmao/OneDrive - UPV EHU/AI_PCMs/NeuralODEs/Oscar/NODE_Oscar_1MLPx3/LearnigData_Rod_ClampedPinned_Rotated_X02_72sols_mode2_revised.mat" # change \ symbol for / , after copy the link from windows
        mf_DataSet = MatFile(filename_DataSet)                # opens a MAT file for reading
        data_DataSet = get_mvariable(mf_DataSet, "DataSet")   # gets a variable and returns an mxArray
        DataSet = jarray(data_DataSet)                        # converts x to a Julia matrix
    
    # Initialize variables
    θ₀_true, κ₀_true = [], []
    px_true, py_true, θ_true, κ_true = Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[]
        
    # Load θ₀ and κ₀ from data 
        for i in 1:72 # Number of trajectories in DataSet: 72
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
        ii = 72 # Number of trajectories included in Learning set: ii of 72 available in DataSet
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
            const _st = st
    
        # Define the ODE
            function ude_dynamics!(du, u, p, s)
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
        Δs = 1/49
        s = Float32.(0:Δs:1)

    # Use the first trajectory's initial state to define prob_nn
        u0 = Float32.(X_ssol_all[1][:, 1])
        prob_nn = ODEProblem(ude_dynamics!, u0, sSpan, p)
    
    # Update the predict function to handle the initial conditions and parameters from the optimizer
    function predict(θ, X_all = X_ssol_all, S = s)
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
        optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))      
        # ADAM loop
        learning_rate = 0.001
        epoch_1 = 1000 # ADAM
        res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(learning_rate), callback = callback, maxiters = epoch_1)
        # LBFGS loop
        epoch_2 = 10000 # LBFGS
        optprob2 = Optimization.OptimizationProblem(optf, res1.u)
        res2 = Optimization.solve(optprob2, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = epoch_2)

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
     c = 36

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
                    Plots.savefig(Pp, "theta_0_sol_180º_mode_2_X02_Theta-s.pdf")
                    display(Pp)
                end
            # κ(s)_true - UDE prediction -
                
            begin
                    Pp = Plots.plot(s, vec(κ_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best)
                    Plots.scatter!(s, κ_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s \,\, [m]", ylabel=L"\dot{\theta}\,\,=\,\,k\,\, [-]")
                    Plots.savefig(Pp, "theta_0_sol_180º_mode_2_X02_Kappa-s.pdf")
                    display(Pp)
                end

            # Phase diagrams comparison
                begin
                    f44 = Plots.plot(vec(θ_true[c]), vec(κ_true[c]), color=[:red :gray], legend=:best , label="Curva original")
                    Plots.scatter!(θ_predict, κ_predict, color=[:black :gray], markersize=3, legend=:best , label="UODE",
                            xlabel=L"θ \,\, [rad]", ylabel=L"κ\,\, [-] ")
                    Plots.savefig(f44, "theta_0_sol_180º_mode_2_X02_phase_C_3MLP.pdf")
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
            Plots.savefig(f55, "theta_0_sol_180º_mode_2_X02_theta-kappa_error.pdf")
            display(f55)
        end

        # Shape of the rod - U=DE prediction vs. Ground Truth 
            # x(s)
                begin
                    Pp = Plots.plot(s, vec(px_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(s, px_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s\,\,[m]", ylabel=L"x\,\,[m]")
                    Plots.savefig(Pp, "theta_0_sol_180º_mode_2_X02_X-s.pdf")
                    display(Pp)
                end
            # y(s)
            begin
                    Pp = Plots.plot(s, vec(py_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(s, py_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s\,\,[m]", ylabel=L"y\,\,[m]")
                    Plots.savefig(Pp, "theta_0_sol_180º_mode_2_X02_Y-s.pdf")
                    display(Pp)
                end
            # (x, y)
                begin
                    f66 = Plots.plot(vec(px_true[c]), vec(py_true[c]), color=[:black :gray], label=["Curva original" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(px_predict, py_predict, color=[:blue :orange], label=["UODE" nothing],
                        xlabel=L"x\,\,[m]", ylabel=L"y\,\,[m]")
                    Plots.savefig(f66, "theta_0_sol_180º_mode_2_X02_X-Y.pdf")
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
# C.- Save NNs

using JLD2

model = U
parameters = p_trained
states = _st

# Save the NN data to a file
@save "NN_NODE_1MLPx3_tanh_X02.jld2" model parameters states
