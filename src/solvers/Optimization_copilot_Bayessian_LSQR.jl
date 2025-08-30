# ---------------------------------------------------------------------------
# Hyperparameter Optimization for Multi-Stage UDE Training with LSQR
# 
# Implements Hyperband and Bayesian Optimization for:
# 1. RAdam → LSQR
# 2. CMA-ES → LSQR  
# 3. DE/PSO → RAdam → LSQR
# ---------------------------------------------------------------------------

# Setup project environment
include("../utils/project_utils.jl")
project_root = setup_project_environment(activate_env = true, instantiate = false)

# Create output directories
plots_dir = joinpath(project_root, "src", "Data", "NODE_HyperOpt_LSQR", "Plots_Results")
mkpath(plots_dir)
println("Results will be saved to: ", plots_dir)

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

# For LSQR
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using FiniteDiff  # For Jacobian computation

# Global optimization
using Evolutionary
using BlackBoxOptim

# Hyperparameter optimization
using Hyperopt
using BayesianOptimization
using GaussianProcesses

# Utilities
using Statistics, Random
using ComponentArrays, Lux, Zygote, StableRNGs
using DataFrames, CSV
using JSON3
using ProgressMeter

# -----------------------------------------------------------------------------
# Load your UDE setup (using code from your file)
println("Loading dataset and setting up UDE...")

using MATLAB
filename_DataSet = joinpath(project_root, "dataset", "LearnigData_Rod_Clamp_Pin_Rot_X02_Y00_mode2_train_085.mat")
println("Loading dataset from: ", filename_DataSet)

if !isfile(filename_DataSet)
    error("Dataset file not found at: $filename_DataSet")
end

mf_DataSet = MatFile(filename_DataSet)
data_DataSet = get_mvariable(mf_DataSet, "DataSet_train")
DataSet = jarray(data_DataSet)

# Dynamically determine the number of trajectories
num_trajectories = size(DataSet, 1)
println("Found $num_trajectories trajectories in the dataset")

# Use subset for hyperopt speed
max_trajectories_to_use = 10  # Adjust based on computational budget
trajectories_to_use = min(max_trajectories_to_use, num_trajectories)
println("Using $trajectories_to_use out of $num_trajectories available trajectories")

# Initialize variables
θ₀_true, κ₀_true = [], []
px_true, py_true, θ_true, κ_true = Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[]

# Load data
for i in 1:trajectories_to_use
    θ₀_i  = Float32.(DataSet[i,3])
    κ₀_i  = Float32.(DataSet[i,112])
    push!(θ₀_true,  θ₀_i)
    push!(κ₀_true,  κ₀_i)
end
θ₀_true = Float32.(collect(θ₀_true))
κ₀_true = Float32.(collect(κ₀_true))

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

# Setup neural network
rng = StableRNG(1111)
m = 4
n = 3
layers = collect([m, 20, 20, 20, n])

# Multilayer FeedForward
const U = Lux.Chain(
    [Dense(fan_in => fan_out, Lux.tanh) for (fan_in, fan_out) in zip(layers[1:end-2], layers[2:end-1])]...,
    Dense(layers[end-1] => layers[end], identity),
) 

# Get the initial parameters and state variables
p_init, st = Lux.setup(rng, U)
p_init = ComponentArray{Float32}(p_init)
const _st = st

# Global counters for NN passes
const NN_FORWARD_COUNT = Ref(0)
const NN_BACKWARD_COUNT = Ref(0)

# Define the ODE
function ude_dynamics!(du, u, p, s)
    NN_FORWARD_COUNT[] += 1
    if any(!isfinite, u)
        du .= 0.0f0
        return nothing
    end
    û_1 = U(u, p, _st)[1] 
    
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
const prob_nn = ODEProblem(ude_dynamics!, u0, sSpan, p_init)

# Prediction function
function predict(θ, X_all = X_ssol_all, S = s)
    try
        [Array(solve(remake(prob_nn, u0 = Float32.(X[:, 1]), tspan = (S[1], S[end]), p = θ),    
                AutoVern7(Rodas5()), saveat = S, abstol = 1e-6, reltol = 1e-6,                  
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))) for X in X_all]
    catch e
        return nothing
    end
end

# Loss function
function loss(θ)
    X̂_sols = predict(θ)
    if X̂_sols === nothing
        return 1e10
    end
    total_loss = sum(mean(abs, X_ssol_all[i] .- X̂_sols[i]) for i in 1:length(X̂_sols))
    return total_loss / length(X̂_sols)
end

# -----------------------------------------------------------------------------
# LSQR IMPLEMENTATION FOR REFINEMENT

"""
    compute_residuals_and_jacobian(θ)
    
Compute residuals and Jacobian for LSQR method
"""
function compute_residuals_and_jacobian(θ; use_finite_diff=true)
    # Compute current predictions
    X̂_sols = predict(θ)
    if X̂_sols === nothing
        return nothing, nothing
    end
    
    # Flatten residuals
    residuals = Float64[]
    for i in 1:length(X̂_sols)
        append!(residuals, vec(X_ssol_all[i] .- X̂_sols[i]))
    end
    
    if use_finite_diff
        # Use finite differences for Jacobian (more stable for ill-conditioned problems)
        residual_func = θ_vec -> begin
            θ_comp = ComponentArray{Float32}(θ_vec, axes(θ))
            X̂_sols_j = predict(θ_comp)
            if X̂_sols_j === nothing
                return fill(1e10, length(residuals))
            end
            res = Float64[]
            for i in 1:length(X̂_sols_j)
                append!(res, vec(X_ssol_all[i] .- X̂_sols_j[i]))
            end
            return res
        end
        
        # Compute Jacobian using central differences
        J = FiniteDiff.finite_difference_jacobian(residual_func, vec(θ))
    else
        # Use automatic differentiation (if preferred)
        # Note: This might be less stable for ill-conditioned problems
        J = Zygote.jacobian(θ -> begin
            X̂_sols_j = predict(θ)
            if X̂_sols_j === nothing
                return fill(1e10, length(residuals))
            end
            res = Float64[]
            for i in 1:length(X̂_sols_j)
                append!(res, vec(X_ssol_all[i] .- X̂_sols_j[i]))
            end
            return res
        end, θ)[1]
    end
    
    return residuals, J
end

"""
    lsqr_refinement(θ_init; hyperparams)
    
Refine parameters using LSQR method
"""
function lsqr_refinement(θ_init; hyperparams=Dict())
    # Extract hyperparameters with defaults
    max_iters = get(hyperparams, :lsqr_maxiter, 100)
    atol = get(hyperparams, :lsqr_atol, 1e-8)
    btol = get(hyperparams, :lsqr_btol, 1e-8)
    λ_reg = get(hyperparams, :lsqr_lambda, 1e-6)
    max_outer_iters = get(hyperparams, :lsqr_outer_iters, 20)
    line_search_alpha = get(hyperparams, :lsqr_alpha, 0.5)
    
    θ_current = copy(θ_init)
    losses_lsqr = Float64[]
    
    for outer_iter in 1:max_outer_iters
        # Compute residuals and Jacobian
        residuals, J = compute_residuals_and_jacobian(θ_current)
        
        if residuals === nothing || J === nothing
            println("  LSQR: Failed to compute residuals/Jacobian at iteration $outer_iter")
            break
        end
        
        current_loss = norm(residuals)^2 / length(residuals)
        push!(losses_lsqr, current_loss)
        
        # Solve using LSQR
        Δθ, stats = lsqr(J, residuals,
                         atol=atol,
                         btol=btol,
                         maxiter=max_iters,
                         λ=λ_reg,
                         verbose=false)
        
        # Line search for stable update
        α = line_search_alpha
        improvement_found = false
        
        for _ in 1:10
            θ_test = θ_current - α * ComponentArray{Float32}(Δθ, axes(θ_current))
            test_loss = loss(θ_test)
            
            if test_loss < current_loss
                θ_current = θ_test
                improvement_found = true
                break
            end
            α *= 0.5
        end
        
        if !improvement_found
            println("  LSQR: No improvement found at iteration $outer_iter, stopping")
            break
        end
        
        # Check convergence
        if outer_iter > 1 && abs(losses_lsqr[end] - losses_lsqr[end-1]) < 1e-6
            println("  LSQR: Converged at iteration $outer_iter")
            break
        end
    end
    
    return θ_current, losses_lsqr
end

# -----------------------------------------------------------------------------
# STRATEGY IMPLEMENTATIONS WITH LSQR

"""
    train_strategy_1(hyperparams; max_iters_dict)
    
Option 1: RAdam → LSQR
"""
function train_strategy_1(hyperparams; max_iters_dict=Dict(:radam=>200, :lsqr=>20))
    p_current = copy(p_init)
    total_time = 0.0
    all_losses = Float64[]
    
    # Phase 1: RAdam
    println("  Strategy 1 - Phase 1: RAdam")
    losses_radam = Float64[]
    callback = function (p, l)
        push!(losses_radam, l)
        push!(all_losses, l)
        return false
    end
    
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, q) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_current)
    
    t1 = @elapsed begin
        res_radam = Optimization.solve(optprob,
                                       Optimisers.RAdam(hyperparams[:radam_lr]),
                                       callback = callback,
                                       maxiters = hyperparams[:radam_iters])
    end
    total_time += t1
    p_current = res_radam.u
    
    # Phase 2: LSQR
    println("  Strategy 1 - Phase 2: LSQR")
    
    lsqr_hyperparams = Dict(
        :lsqr_maxiter => hyperparams[:lsqr_maxiter],
        :lsqr_atol => hyperparams[:lsqr_atol],
        :lsqr_btol => hyperparams[:lsqr_btol],
        :lsqr_lambda => hyperparams[:lsqr_lambda],
        :lsqr_outer_iters => max_iters_dict[:lsqr],
        :lsqr_alpha => hyperparams[:lsqr_alpha]
    )
    
    t2 = @elapsed begin
        p_current, losses_lsqr = lsqr_refinement(p_current; hyperparams=lsqr_hyperparams)
        append!(all_losses, losses_lsqr)
    end
    total_time += t2
    
    final_loss = loss(p_current)
    
    return final_loss, p_current, all_losses, total_time
end

"""
    train_strategy_2(hyperparams; max_iters_dict)
    
Option 2: CMA-ES → LSQR
"""
function train_strategy_2(hyperparams; max_iters_dict=Dict(:cmaes=>100, :lsqr=>20))
    p_current = copy(p_init)
    total_time = 0.0
    all_losses = Float64[]
    
    # Phase 1: CMA-ES
    println("  Strategy 2 - Phase 1: CMA-ES")
    
    function loss_vec(θ_vec)
        θ_comp = ComponentArray{Float32}(θ_vec, axes(p_current))
        return loss(θ_comp)
    end
    
    lower_bounds = fill(-5.0, length(vec(p_current)))
    upper_bounds = fill(5.0, length(vec(p_current)))
    
    n_params = length(vec(p_current))
    if hyperparams[:cmaes_popsize] == "auto"
        μ = div(n_params, 4)
        λ = 2 * μ
    else
        λ = hyperparams[:cmaes_popsize]
        μ = div(λ, 2)
    end
    
    t1 = @elapsed begin
        result_cmaes = Evolutionary.optimize(
            loss_vec,
            BoxConstraints(lower_bounds, upper_bounds),
            CMAES(μ = μ, λ = λ),
            Evolutionary.Options(
                iterations = hyperparams[:cmaes_iters],
                show_trace = false,
                store_trace = true
            )
        )
        p_current = ComponentArray{Float32}(Evolutionary.minimizer(result_cmaes), axes(p_current))
        
        # Extract losses from trace
        for state in result_cmaes.trace
            push!(all_losses, state.value)
        end
    end
    total_time += t1
    
    # Phase 2: LSQR
    println("  Strategy 2 - Phase 2: LSQR")
    
    lsqr_hyperparams = Dict(
        :lsqr_maxiter => hyperparams[:lsqr_maxiter],
        :lsqr_atol => hyperparams[:lsqr_atol],
        :lsqr_btol => hyperparams[:lsqr_btol],
        :lsqr_lambda => hyperparams[:lsqr_lambda],
        :lsqr_outer_iters => max_iters_dict[:lsqr],
        :lsqr_alpha => hyperparams[:lsqr_alpha]
    )
    
    t2 = @elapsed begin
        p_current, losses_lsqr = lsqr_refinement(p_current; hyperparams=lsqr_hyperparams)
        append!(all_losses, losses_lsqr)
    end
    total_time += t2
    
    final_loss = loss(p_current)
    
    return final_loss, p_current, all_losses, total_time
end

"""
    train_strategy_3(hyperparams; max_iters_dict)
    
Option 3: DE/PSO → RAdam → LSQR
"""
function train_strategy_3(hyperparams; max_iters_dict=Dict(:global=>50, :radam=>200, :lsqr=>20))
    p_current = copy(p_init)
    total_time = 0.0
    all_losses = Float64[]
    
    # Phase 1: DE or PSO
    function loss_vec(θ_vec)
        θ_comp = ComponentArray{Float32}(θ_vec, axes(p_current))
        return loss(θ_comp)
    end
    
    lower_bounds = fill(-5.0, length(vec(p_current)))
    upper_bounds = fill(5.0, length(vec(p_current)))
    
    if hyperparams[:use_de]
        println("  Strategy 3 - Phase 1: Differential Evolution")
        t1 = @elapsed begin
            result_de = Evolutionary.optimize(
                loss_vec,
                BoxConstraints(lower_bounds, upper_bounds),
                DE(populationSize = hyperparams[:de_popsize],
                   F = hyperparams[:de_f],
                   CR = hyperparams[:de_cr]),
                Evolutionary.Options(
                    iterations = hyperparams[:global_iters],
                    show_trace = false,
                    store_trace = true
                )
            )
            p_current = ComponentArray{Float32}(Evolutionary.minimizer(result_de), axes(p_current))
            for state in result_de.trace
                push!(all_losses, state.value)
            end
        end
    else
        println("  Strategy 3 - Phase 1: Particle Swarm")
        t1 = @elapsed begin
            result_pso = Evolutionary.optimize(
                loss_vec,
                BoxConstraints(lower_bounds, upper_bounds),
                PSO(n_particles = hyperparams[:pso_particles],
                    w = hyperparams[:pso_w],
                    c1 = hyperparams[:pso_c1],
                    c2 = hyperparams[:pso_c2]),
                Evolutionary.Options(
                    iterations = hyperparams[:global_iters],
                    show_trace = false,
                    store_trace = true
                )
            )
            p_current = ComponentArray{Float32}(Evolutionary.minimizer(result_pso), axes(p_current))
            for state in result_pso.trace
                push!(all_losses, state.value)
            end
        end
    end
    total_time += t1
    
    # Phase 2: RAdam
    println("  Strategy 3 - Phase 2: RAdam")
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, q) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_current)
    
    losses_radam = Float64[]
    callback = function (p, l)
        push!(losses_radam, l)
        push!(all_losses, l)
        return false
    end
    
    t2 = @elapsed begin
        res_radam = Optimization.solve(optprob,
                                       Optimisers.RAdam(hyperparams[:radam_lr]),
                                       callback = callback,
                                       maxiters = hyperparams[:radam_iters])
    end
    total_time += t2
    p_current = res_radam.u
    
    # Phase 3: LSQR
    println("  Strategy 3 - Phase 3: LSQR")
    
    lsqr_hyperparams = Dict(
        :lsqr_maxiter => hyperparams[:lsqr_maxiter],
        :lsqr_atol => hyperparams[:lsqr_atol],
        :lsqr_btol => hyperparams[:lsqr_btol],
        :lsqr_lambda => hyperparams[:lsqr_lambda],
        :lsqr_outer_iters => max_iters_dict[:lsqr],
        :lsqr_alpha => hyperparams[:lsqr_alpha]
    )
    
    t3 = @elapsed begin
        p_current, losses_lsqr = lsqr_refinement(p_current; hyperparams=lsqr_hyperparams)
        append!(all_losses, losses_lsqr)
    end
    total_time += t3
    
    final_loss = loss(p_current)
    
    return final_loss, p_current, all_losses, total_time
end

# -----------------------------------------------------------------------------
# HYPERBAND IMPLEMENTATION

println("\n" * "="^60)
println("HYPERBAND OPTIMIZATION WITH LSQR")
println("="^60)

function hyperband_for_strategy(strategy_fn, hyperspace, strategy_name; 
                                max_resource=400, η=3)
    """
    Hyperband for a given strategy with LSQR
    """
    
    R = max_resource
    s_max = floor(Int, log(R) / log(η))
    B = (s_max + 1) * R
    
    best_config = nothing
    best_loss = Inf
    all_results = []
    
    println("Running Hyperband for $strategy_name")
    println("s_max = $s_max, B = $B")
    
    for s in s_max:-1:0
        println("\nBracket s = $s")
        n = ceil(Int, B / R / (s + 1) * η^s / (η^s))
        r = R * η^(-s)
        
        println("Initial configs: n = $n, initial resource: r = $r")
        
        # Generate n random configurations
        configs = []
        for i in 1:n
            config = Dict{Symbol, Any}()
            for (param_name, param_range) in hyperspace
                if param_range isa Tuple{Float64, Float64}
                    config[param_name] = param_range[1] + (param_range[2] - param_range[1]) * rand()
                elseif param_range isa Tuple{Int, Int}
                    config[param_name] = rand(param_range[1]:param_range[2])
                elseif param_range isa Vector
                    config[param_name] = rand(param_range)
                elseif param_range isa Tuple{String, String} && param_range[1] == "log"
                    low, high = log(parse(Float64, split(param_range[2], ":")[1])), 
                               log(parse(Float64, split(param_range[2], ":")[2]))
                    config[param_name] = exp(low + (high - low) * rand())
                end
            end
            push!(configs, config)
        end
        
        # Successive halving
        for i in 0:s
            n_i = floor(Int, n * η^(-i))
            r_i = floor(Int, r * η^i)
            
            println("  Round $i: evaluating $(min(n_i, length(configs))) configs with resource $r_i")
            
            # Evaluate configurations
            results = []
            for (idx, config) in enumerate(configs[1:min(n_i, length(configs))])
                # Adjust iterations based on resource
                if haskey(config, :radam_iters)
                    config[:radam_iters] = min(config[:radam_iters], r_i)
                end
                if haskey(config, :cmaes_iters)
                    config[:cmaes_iters] = min(config[:cmaes_iters], r_i)
                end
                if haskey(config, :global_iters)
                    config[:global_iters] = min(config[:global_iters], div(r_i, 3))
                    config[:radam_iters] = min(config[:radam_iters], div(r_i, 2))
                end
                
                try
                    loss_val, _, _, time = strategy_fn(config)
                    push!(results, (config=config, loss=loss_val, time=time))
                    println("    Config $idx: loss = $(round(loss_val, digits=6))")
                catch e
                    println("    Config $idx failed: $e")
                    push!(results, (config=config, loss=1e10, time=0.0))
                end
            end
            
            # Keep best configurations
            sort!(results, by=x->x.loss)
            n_keep = max(1, floor(Int, n_i/η))
            configs = [r.config for r in results[1:min(n_keep, length(results))]]
            
            # Track best
            if length(results) > 0 && results[1].loss < best_loss
                best_loss = results[1].loss
                best_config = results[1].config
            end
            
            push!(all_results, results)
        end
    end
    
    return best_config, best_loss, all_results
end

# Define hyperparameter spaces for LSQR-based strategies
hyperspace_strategy1 = [
    :radam_lr => ("log", "1e-5:1e-1"),
    :radam_iters => (100, 500),
    :lsqr_maxiter => (50, 200),
    :lsqr_atol => ("log", "1e-10:1e-6"),
    :lsqr_btol => ("log", "1e-10:1e-6"),
    :lsqr_lambda => ("log", "1e-8:1e-3"),
    :lsqr_alpha => (0.1, 1.0)
]

hyperspace_strategy2 = [
    :cmaes_popsize => ["auto", 20, 50, 100],
    :cmaes_iters => (50, 200),
    :lsqr_maxiter => (50, 200),
    :lsqr_atol => ("log", "1e-10:1e-6"),
    :lsqr_btol => ("log", "1e-10:1e-6"),
    :lsqr_lambda => ("log", "1e-8:1e-3"),
    :lsqr_alpha => (0.1, 1.0)
]

hyperspace_strategy3 = [
    :use_de => [true, false],
    :de_popsize => (20, 100),
    :de_f => (0.5, 1.0),
    :de_cr => (0.7, 1.0),
    :pso_particles => (20, 50),
    :pso_w => (0.4, 0.9),
    :pso_c1 => (1.0, 2.0),
    :pso_c2 => (1.0, 2.0),
    :global_iters => (30, 100),
    :radam_lr => ("log", "1e-5:1e-1"),
    :radam_iters => (100, 300),
    :lsqr_maxiter => (50, 200),
    :lsqr_atol => ("log", "1e-10:1e-6"),
    :lsqr_btol => ("log", "1e-10:1e-6"),
    :lsqr_lambda => ("log", "1e-8:1e-3"),
    :lsqr_alpha => (0.1, 1.0)
]

# Run Hyperband for each strategy
hyperband_results = Dict()

println("\n" * "="^30 * " STRATEGY 1 " * "="^30)
best_config1, best_loss1, results1 = hyperband_for_strategy(
    train_strategy_1, hyperspace_strategy1, "Strategy 1: RAdam → LSQR",
    max_resource=300, η=3
)
hyperband_results["Strategy1"] = (config=best_config1, loss=best_loss1)

println("\n" * "="^30 * " STRATEGY 2 " * "="^30)
best_config2, best_loss2, results2 = hyperband_for_strategy(
    train_strategy_2, hyperspace_strategy2, "Strategy 2: CMA-ES → LSQR",
    max_resource=250, η=3
)
hyperband_results["Strategy2"] = (config=best_config2, loss=best_loss2)

println("\n" * "="^30 * " STRATEGY 3 " * "="^30)
best_config3, best_loss3, results3 = hyperband_for_strategy(
    train_strategy_3, hyperspace_strategy3, "Strategy 3: DE/PSO → RAdam → LSQR",
    max_resource=400, η=3
)
hyperband_results["Strategy3"] = (config=best_config3, loss=best_loss3)

# -----------------------------------------------------------------------------
# BAYESIAN OPTIMIZATION IMPLEMENTATION

println("\n" * "="^60)
println("BAYESIAN OPTIMIZATION WITH LSQR")
println("="^60)

using GaussianProcesses
using Optim

function bayesian_optimization_strategy(strategy_fn, bounds_dict, strategy_name;
                                       n_initial=5, n_iterations=20)
    
    println("Running Bayesian Optimization for $strategy_name")
    
    # Convert bounds to arrays
    param_names = collect(keys(bounds_dict))
    lower_bounds = Float64[]
    upper_bounds = Float64[]
    log_scale = Bool[]
    
    for (param, bound) in bounds_dict
        if bound[1] == "log"
            push!(lower_bounds, log(bound[2]))
            push!(upper_bounds, log(bound[3]))
            push!(log_scale, true)
        else
            push!(lower_bounds, bound[1])
            push!(upper_bounds, bound[2])
            push!(log_scale, false)
        end
    end
    
    n_dims = length(param_names)
    
    # Initial random sampling
    X_observed = zeros(n_dims, 0)
    y_observed = Float64[]
    
    println("Initial random sampling ($n_initial points)...")
    for i in 1:n_initial
        # Sample random point
        x = zeros(n_dims)
        for j in 1:n_dims
            x[j] = lower_bounds[j] + (upper_bounds[j] - lower_bounds[j]) * rand()
        end
        
        # Convert to hyperparameters
        config = Dict{Symbol, Any}()
        for (j, param) in enumerate(param_names)
            if log_scale[j]
                config[param] = exp(x[j])
            else
                config[param] = x[j]
            end
            
            # Type conversions
            if param in [:radam_iters, :cmaes_iters, :global_iters, :lsqr_maxiter,
                        :de_popsize, :pso_particles]
                config[param] = round(Int, config[param])
            end
            if param == :use_de
                config[param] = config[param] > 0.5
            end
        end
        
        # Evaluate
        try
            loss_val, _, _, _ = strategy_fn(config)
            X_observed = hcat(X_observed, x)
            push!(y_observed, loss_val)
            println("  Sample $i: loss = $(round(loss_val, digits=6))")
        catch e
            println("  Sample $i failed: $e")
            X_observed = hcat(X_observed, x)
            push!(y_observed, 1e10)
        end
    end
    
    # Bayesian optimization loop
    for iter in 1:n_iterations
        println("\nBO Iteration $iter/$n_iterations")
        
        # Fit Gaussian Process
        y_mean = mean(y_observed)
        y_std = std(y_observed) + 1e-6
        y_normalized = (y_observed .- y_mean) ./ y_std
        
        # Create GP with Matern kernel
        kern = Mat52Iso(1.0, 1.0)
        gp = GP(X_observed', y_normalized, MeanConst(0.0), kern, -2.0)
        
        # Optimize GP hyperparameters
        optimize!(gp)
        
        # Acquisition function: Expected Improvement
        best_y = minimum(y_normalized)
        
        function acquisition(x)
            μ, σ² = predict_f(gp, reshape(x, :, 1))
            σ = sqrt(σ²[1] + 1e-6)
            
            # Expected Improvement
            if σ > 0
                Z = (best_y - μ[1]) / σ
                Φ = cdf(Normal(), Z)
                φ = pdf(Normal(), Z)
                ei = σ * (Z * Φ + φ)
                return -ei  # Minimize negative EI
            else
                return 0.0
            end
        end
        
        # Find next point
        result = optimize(acquisition, lower_bounds, upper_bounds, 
                         lower_bounds .+ (upper_bounds .- lower_bounds) .* rand(n_dims),
                         Fminbox(LBFGS()), 
                         Optim.Options(iterations=100))
        x_next = Optim.minimizer(result)
        
        # Convert and evaluate
        config = Dict{Symbol, Any}()
        for (j, param) in enumerate(param_names)
            if log_scale[j]
                config[param] = exp(x_next[j])
            else
                config[param] = x_next[j]
            end
            
            # Type conversions
            if param in [:radam_iters, :cmaes_iters, :global_iters, :lsqr_maxiter,
                        :de_popsize, :pso_particles]
                config[param] = round(Int, config[param])
            end
            if param == :use_de
                config[param] = config[param] > 0.5
            end
        end
        
        # Evaluate
        try
            loss_val, _, _, _ = strategy_fn(config)
            X_observed = hcat(X_observed, x_next)
            push!(y_observed, loss_val)
            println("  New sample: loss = $(round(loss_val, digits=6))")
            
            if loss_val < minimum(y_observed[1:end-1])
                println("  🎯 New best configuration found!")
            end
        catch e
            println("  Evaluation failed: $e")
            X_observed = hcat(X_observed, x_next)
            push!(y_observed, 1e10)
        end
    end
    
    # Return best configuration
    best_idx = argmin(y_observed)
    best_x = X_observed[:, best_idx]
    
    best_config = Dict{Symbol, Any}()
    for (j, param) in enumerate(param_names)
        if log_scale[j]
            best_config[param] = exp(best_x[j])
        else
            best_config[param] = best_x[j]
        end
        
        # Type conversions
        if param in [:radam_iters, :cmaes_iters, :global_iters, :lsqr_maxiter,
                    :de_popsize, :pso_particles]
            best_config[param] = round(Int, best_config[param])
        end
        if param == :use_de
            best_config[param] = best_config[param] > 0.5
        end
    end
    
    return best_config, y_observed[best_idx], X_observed, y_observed
end

# Define bounds for Bayesian Optimization with LSQR
bo_bounds_strategy1 = Dict(
    :radam_lr => ("log", 1e-5, 1e-1),
    :radam_iters => ("linear", 100.0, 500.0),
    :lsqr_maxiter => ("linear", 50.0, 200.0),
    :lsqr_atol => ("log", 1e-10, 1e-6),
    :lsqr_btol => ("log", 1e-10, 1e-6),
    :lsqr_lambda => ("log", 1e-8, 1e-3),
    :lsqr_alpha => ("linear", 0.1, 1.0)
)

bo_bounds_strategy2 = Dict(
    :cmaes_popsize => ("linear", 20.0, 100.0),
    :cmaes_iters => ("linear", 50.0, 200.0),
    :lsqr_maxiter => ("linear", 50.0, 200.0),
    :lsqr_atol => ("log", 1e-10, 1e-6),
    :lsqr_btol => ("log", 1e-10, 1e-6),
    :lsqr_lambda => ("log", 1e-8, 1e-3),
    :lsqr_alpha => ("linear", 0.1, 1.0)
)

bo_bounds_strategy3 = Dict(
    :use_de => ("linear", 0.0, 1.0),
    :de_popsize => ("linear", 20.0, 100.0),
    :de_f => ("linear", 0.5, 1.0),
    :de_cr => ("linear", 0.7, 1.0),
    :pso_particles => ("linear", 20.0, 50.0),
    :pso_w => ("linear", 0.4, 0.9),
    :global_iters => ("linear", 30.0, 100.0),
    :radam_lr => ("log", 1e-5, 1e-1),
    :radam_iters => ("linear", 100.0, 300.0),
    :lsqr_maxiter => ("linear", 50.0, 200.0),
    :lsqr_atol => ("log", 1e-10, 1e-6),
    :lsqr_btol => ("log", 1e-10, 1e-6),
    :lsqr_lambda => ("log", 1e-8, 1e-3),
    :lsqr_alpha => ("linear", 0.1, 1.0)
)

# Run Bayesian Optimization
bo_results = Dict()

println("\n" * "="^30 * " BO STRATEGY 1 " * "="^30)
bo_config1, bo_loss1, _, _ = bayesian_optimization_strategy(
    train_strategy_1, bo_bounds_strategy1, 
    "Strategy 1: RAdam → LSQR",
    n_initial=5, n_iterations=15
)
bo_results["Strategy1"] = (config=bo_config1, loss=bo_loss1)

println("\n" * "="^30 * " BO STRATEGY 2 " * "="^30)
bo_config2, bo_loss2, _, _ = bayesian_optimization_strategy(
    train_strategy_2, bo_bounds_strategy2,
    "Strategy 2: CMA-ES → LSQR",
    n_initial=5, n_iterations=15
)
bo_results["Strategy2"] = (config=bo_config2, loss=bo_loss2)

println("\n" * "="^30 * " BO STRATEGY 3 " * "="^30)
bo_config3, bo_loss3, _, _ = bayesian_optimization_strategy(
    train_strategy_3, bo_bounds_strategy3,
    "Strategy 3: DE/PSO → RAdam → LSQR",
    n_initial=5, n_iterations=15
)
bo_results["Strategy3"] = (config=bo_config3, loss=bo_loss3)

# -----------------------------------------------------------------------------
# FINAL COMPARISON

println("\n" * "="^60)
println("FINAL RESULTS - LSQR STRATEGIES")
println("="^60)

# Create comparison table
comparison_df = DataFrame(
    Strategy = String[],
    Method = String[],
    BestLoss = Float64[],
    Time = String[]
)

for (strategy_name, result) in hyperband_results
    push!(comparison_df, (
        strategy_name, 
        "Hyperband", 
        result.loss,
        "N/A"
    ))
end

for (strategy_name, result) in bo_results
    push!(comparison_df, (
        strategy_name, 
        "Bayesian Opt", 
        result.loss,
        "N/A"
    ))
end

sort!(comparison_df, :BestLoss)
println(comparison_df)

# Save results
best_configs = Dict(
    "Hyperband" => hyperband_results,
    "BayesianOpt" => bo_results,
    "Comparison" => comparison_df
)

open(joinpath(plots_dir, "best_hyperparameters_lsqr.json"), "w") do io
    JSON3.write(io, best_configs)
end

# Print best overall configuration
overall_best = comparison_df[1, :]
println("\n" * "="^60)
println("🏆 BEST CONFIGURATION (LSQR-based)")
println("="^60)
println("Strategy: $(overall_best.Strategy)")
println("Method: $(overall_best.Method)")
println("Loss: $(overall_best.BestLoss)")

# Retrieve and display best configuration details
best_strategy = overall_best.Strategy
best_method = overall_best.Method

if best_method == "Hyperband"
    best_config = hyperband_results[best_strategy].config
else
    best_config = bo_results[best_strategy].config
end

println("\nBest Hyperparameters:")
for (key, value) in best_config
    println("  $key: $value")
end

println("="^60)
println("\n✅ Hyperparameter optimization with LSQR complete!")
println("Results saved to: $plots_dir")

# Plot comparison
pl_comparison = plot(
    title="LSQR-based Strategy Optimization",
    xlabel="Strategy",
    ylabel="Best Loss",
    yaxis=:log10,
    legend=:topright
)

strategies = ["Strategy1", "Strategy2", "Strategy3"]
hb_losses = [hyperband_results[s].loss for s in strategies]
bo_losses = [bo_results[s].loss for s in strategies]

bar!(pl_comparison, strategies, hb_losses, label="Hyperband", alpha=0.7, color=:blue)
bar!(pl_comparison, strategies, bo_losses, label="Bayesian Opt", alpha=0.7, color=:red)

savefig(pl_comparison, joinpath(plots_dir, "lsqr_strategies_comparison.pdf"))
display(pl_comparison)