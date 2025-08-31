# ---------------------------------------------------------------------------
"
    Neural ODE approximation to the Kinematics of a Clamped-pinned Cosserat beam.
    Solution of the Inverse Problem from a home position as the Clamping is rotated.
    07-02-2025 (revised and modified 31-08-2025)
    University of the Basque Country - EHU
    José Luis Ruiz Erezuma & Oscar Altuzarra
"
# ---------------------------------------------------------------------------
# Setup project environment and configuration
    include("../utils/project_utils.jl")

    # Setup project environment
    project_root = setup_project_environment(activate_env = true, instantiate = false)

    using Dates
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    # Create output directories for plots and results
    plots_dir = joinpath(project_root, "src", "Data", timestamp)
    mkpath(plots_dir)
    println("Plots & Results will be saved to: ", plots_dir)

# -----------------------------------------------------------------------------
# 0 .- Load Packages

    using Lux 
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

        # Use relative path based on project root for repository portability
        #filename_DataSet = joinpath(project_root, "dataset", "LearnigData_Rod_Clamp_Pin_Rot_X05_Y00_72sols_mode2.mat")
        #filename_DataSet = joinpath(project_root, "dataset", "LearnigData_Rod_Clamp_Pin_Rot_X05_Y00_mode2_train_085.mat")
        filename_DataSet = joinpath(project_root, "dataset", "LearnigData_Rod_Clamp_Pin_Rot_X05_Y00_mode2_test_015.mat")
        println("Loading dataset from: ", filename_DataSet)
        
        # Verify file exists before attempting to load
        if !isfile(filename_DataSet)
            error("Dataset file not found at: $filename_DataSet")
        end
        
        mf_DataSet = MatFile(filename_DataSet)                # opens a Matlab file for reading
        #data_DataSet = get_mvariable(mf_DataSet, "DataSet_temp")
        #data_DataSet = get_mvariable(mf_DataSet, "DataSet_train")   # gets a variable and returns an mxArray
        data_DataSet = get_mvariable(mf_DataSet, "DataSet_test")
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

        # Calculate the number of integration points from the actual arrays
        int_points = length(px_true[1])  # Get length from the first trajectory's position array
        println("Number of integration points: ", int_points)

    # Combine all ii trajectories into a single Learning dataset
        X_ssol_all = [hcat(px_true[i], py_true[i], θ_true[i], κ_true[i])' for i in 1:ii]
            
# ----------------------------------------------------------------------------
# B.- Load the UODE

    # Load the data and the MLPs structure from the file
    jld2_file = joinpath(project_root, "src", "Data", "NN_NODE_1MLPx3_tanh_X05_R1.jld2")
    using JLD2
    """
        load_NN(jld2_file)

    Load a neural network model from a JLD2 file, handling ReconstructedStatic compatibility issues.
    Returns (model, parameters, states) tuple.
    """
    function load_NN(jld2_file)
        # Verify file exists
        if !isfile(jld2_file)
            error("JLD2 file not found at: $jld2_file")
        end

        println("Loading neural network model from: ", jld2_file)

        # Load the neural network directly and reconstruct if needed

        model, parameters, states = try
            println("Attempting direct load...")
            data = load(jld2_file)
            
            # Check if we need to reconstruct the model
            loaded_model = data["model"]
            model_str = string(typeof(loaded_model))
            
            if contains(model_str, "Chain{JLD2.ReconstructedStatic") || contains(model_str, "ReconstructedStatic")
                println("Detected JLD2.ReconstructedStatic issue - reconstructing model from actual structure...")
                
                # Analyze the actual model structure
                println("Analyzing model structure...")
                
                # Get the layers - they are stored as a ReconstructedStatic NamedTuple
                layers_nt = loaded_model.layers
                println("Layers type: ", typeof(layers_nt))
                
                # For ReconstructedStatic, we need to access the fields and data differently
                # Let's inspect what fields are available
                println("Available fields in layers_nt: ", fieldnames(typeof(layers_nt)))
                
                # Try to access the data directly from the ReconstructedStatic
                # Based on the type signature, the data should be a tuple of layers
                new_layers = []
                try
                    # For ReconstructedStatic, the actual data is in the last type parameter
                    # From the type info: (:layer_1, :layer_2, :layer_3, :layer_4) with 
                    # Tuple{Dense1, Dense2, Dense3, Dense4} data
                    
                    # Let's try to get the actual layers from the ReconstructedStatic
                    layers_data = if hasfield(typeof(layers_nt), :data)
                        layers_nt.data
                    else
                        # Alternative: try to reconstruct based on known structure
                        (getproperty(layers_nt, :layer_1), getproperty(layers_nt, :layer_2), 
                        getproperty(layers_nt, :layer_3), getproperty(layers_nt, :layer_4))
                    end
                    
                    num_layers = length(layers_data)
                    println("Number of layers: ", num_layers)
                    
                    # Build the new model based on the actual structure
                    for (i, layer) in enumerate(layers_data)
                        layer_name = [:layer_1, :layer_2, :layer_3, :layer_4][i]
                        println("Processing $layer_name...")
                        println("Layer $i ($layer_name): ", typeof(layer))
                        
                        # Extract dimensions and activation from the ReconstructedStatic layer
                        # The layer should have the structure from the warning message
                        activation = getproperty(layer, :activation)
                        in_dim = getproperty(layer, :in_dims)
                        out_dim = getproperty(layer, :out_dims)
                        
                        println("  Dimensions: $in_dim → $out_dim, Activation: $activation")
                        
                        # Create new layer with proper Lux types
                        new_layer = Dense(in_dim => out_dim, activation)
                        push!(new_layers, new_layer)
                    end
                    
                catch e3
                    println("Failed to access layers data: $e3")

                end
                
                # Create the new model with extracted architecture
                if length(new_layers) > 0
                    reconstructed_model = Chain(new_layers...)
                    println("✓ Model reconstructed from actual structure with $(length(new_layers)) layers")
                else
                    error("Could not extract any layer information from the loaded model")
                end
                
                reconstructed_model, data["parameters"], data["states"]
            else
                # Model loaded correctly
                data["model"], data["parameters"], data["states"]
            end
        catch e
            println("Direct load failed with error: ", e)
            error("Failed to load neural network model. Please check the JLD2 file format and compatibility.")
        end

        println("✓ Neural network model loaded successfully")
        return model, parameters, states
    end

    # Load the neural network using the function
    model, parameters, states = load_NN(jld2_file)

    # Use the exact same assignment pattern as the working file
    const U = model
    p = parameters  
    const _st = states

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
        prob_nn = ODEProblem(ude_dynamics!, u0, sSpan, p) #ODEProblem(nn_dynamics!, u0, sSpan, p)

# ----------------------------------------------------------------------------
# C.- MAKING Prediction (UODE):
     # Generate a New Initial state vector {x₀, y₀, θ₀, κ(θ₀)}

     x₀ = [0 for i in 1:ii]
     y₀ = [0 for i in 1:ii]
     
     X_ssssol = [hcat(x₀[i], y₀[i], θ₀_true[i], κ₀_true[i])' for i in 1:ii]
    
     # Choose rod pose to test comparison
     c = 1

     # Convert the 4x1 Adjoint matrix to a Vector{Float32}
        new_initial_state = vec(X_ssssol[c])

        new_initial_state = [new_initial_state[1]; new_initial_state[2]; new_initial_state[3]; new_initial_state[4]]

    # Predict using the trained parameters
        new_prediction = Array(solve(remake(prob_nn, u0 = new_initial_state, p = p), 
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
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_Theta-s.pdf"))
                    display(Pp)
                end
            # κ(s)_true - UDE prediction -
                
            begin
                    Pp = Plots.plot(s, vec(κ_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best)
                    Plots.scatter!(s, κ_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s \,\, [m]", ylabel=L"\dot{\theta}\,\,=\,\,k\,\, [-]")
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_Kappa-s.pdf"))
                    display(Pp)
                end

            # Phase diagrams comparison
                begin
                    f44 = Plots.plot(vec(θ_true[c]), vec(κ_true[c]), color=[:red :gray], legend=:best , label="Curva original")
                    Plots.scatter!(θ_predict, κ_predict, color=[:black :gray], markersize=3, legend=:best , label="UODE",
                            xlabel=L"θ \,\, [rad]", ylabel=L"κ\,\, [-] ")
                    Plots.savefig(f44, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_phase_C_3MLP.pdf"))
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
                Plots.savefig(f55, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_theta-kappa_error.pdf"))
                display(f55)
            end

        # Shape of the rod - U=DE prediction vs. Ground Truth 
            # x(s)
                begin
                    Pp = Plots.plot(s, vec(px_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(s, px_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s\,\,[m]", ylabel=L"x\,\,[m]")
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_X-s.pdf"))
                    display(Pp)
                end
            # y(s)
            begin
                    Pp = Plots.plot(s, vec(py_true[c]), color=[:black :gray], label=["Training Data" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(s, py_predict, color=[:red :orange], label=["UODE Approximation" nothing],
                        xlabel=L"s\,\,[m]", ylabel=L"y\,\,[m]")
                    Plots.savefig(Pp, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_Y-s.pdf"))
                    display(Pp)
                end
            # (x, y)
                begin
                    f66 = Plots.plot(vec(px_true[c]), vec(py_true[c]), color=[:black :gray], label=["Curva original" nothing], legend=:best, aspect_ratio=:equal)
                    Plots.scatter!(px_predict, py_predict, color=[:blue :orange], label=["UODE" nothing],
                        xlabel=L"x\,\,[m]", ylabel=L"y\,\,[m]")
                    Plots.savefig(f66, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_X-Y.pdf"))
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
                    Plots.savefig(f77, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_X-Y_error.pdf"))
                    display(f77)
                end

        # Final plots
            # θ(s), κ(s) 
                begin
                f44_55 = Plots.plot(f44, f55, layout = (1, 2), size=(1200, 500), left_margin=10Plots.mm, right_margin=10Plots.mm, top_margin=10Plots.mm, bottom_margin=10Plots.mm)
                Plots.savefig(f44_55, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_theta-kappa_combined.pdf"))
                display(f44_55)
                end
            # x(s), y(s)
                begin
                f66_77 = Plots.plot(f66, f77, layout = (1, 2), size=(1200, 500), left_margin=10Plots.mm, right_margin=10Plots.mm, top_margin=10Plots.mm, bottom_margin=10Plots.mm)
                Plots.savefig(f66_77, joinpath(plots_dir, "theta_0_sol_270º_mode_2_X02_X-Y_combined.pdf"))
                display(f66_77)
                end

# ----------------------------------------------------------------------------
# D.- MAKING error comparison True vs Prediction:

    using CairoMakie

    X = repeat(θ₀_true, inner = int_points)  # Repite cada θ₀_i 50 veces (uno por cada pX_i)
    Y = repeat(s, outer = trajectories_to_use)  # Índices de 1 a 50, repetidos para cada θ₀
    Z_param_x = vcat(px_true...)  # Aplanamos la lista de listas de px en un solo vector
    Z_param_y = vcat(py_true...)  # Aplanamos la lista de listas de py en un solo vector

    # Choose rod poses to test comparison
    px_predict_all, py_predict_all, θ_predict_all, κ_predict_all = Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[]

    for j in 1:trajectories_to_use
    # Convert the 4x1 Adjoint matrix to a Vector{Float32}
    new_initial_state = vec(X_ssssol[j])
    new_initial_state_j = [new_initial_state[1]; new_initial_state[2]; new_initial_state[3]; new_initial_state[4]]

    # Predict using the trained parameters
    new_prediction_j = Array(solve(remake(prob_nn, u0 = new_initial_state_j, p = p), 
                                AutoVern7(Rodas5()), saveat = s, abstol = 1e-6, reltol = 1e-6, 
                                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))

    px_predict_j = new_prediction_j[1,1:end]
    py_predict_j = new_prediction_j[2,1:end]
    θ_predict_j  = new_prediction_j[3,1:end]
    κ_predict_j  = new_prediction_j[4,1:end]
    # Append to lists
    push!(px_predict_all, px_predict_j)
    push!(py_predict_all, py_predict_j)
    push!(θ_predict_all,  θ_predict_j)
    push!(κ_predict_all,  κ_predict_j)
    end
    Z_predict_x = vcat(px_predict_all...)  # Aplanamos la lista de listas de px en un solo vector
    Z_predict_y = vcat(py_predict_all...)  # Aplanamos la lista de listas de py en un solo vector

    # Plot Error on Data x and y
    ABSError_x_y = sqrt.((Z_predict_x .- Z_param_x ).^2 .+ (Z_predict_y .- Z_param_y ).^2)
    begin
        fig = Figure()
        ax = CairoMakie.Axis3(fig[1, 1], azimuth = -0.4 * pi, aspect = (1,1,1), xlabel="θ₀ [rad]", ylabel="s [m]", zlabel="Error_ABS_x_y [m]", title =" Neural Network Predicted_x_y Error on Test-DataSet")
        sABSError = CairoMakie.scatter!(ax, X, Y, ABSError_x_y; markersize=6, alpha=1,
        color=:orange, strokecolor=:black, strokewidth=0.5)
        #axislegend(ax, [sABSError], ["Error_ABS_x_y"])
        display(fig)
        save(joinpath(plots_dir, "Error_on_Data_set_vs_Predicted_x_y.pdf"), fig, pdf_version="1.4")  
    end    