# ---------------------------------------------------------------------------
# Hyperparameter Optimization for Multi-Stage UDE Training Strategies
# 
# Implements Hyperband and Bayesian Optimization for:
# 1. RAdam → LBFGS/LM
# 2. CMA-ES → LBFGS/LM  
# 3. DE/PSO → RAdam → LBFGS/LM
# ---------------------------------------------------------------------------

# Setup project environment
include("../utils/project_utils.jl")
project_root = setup_project_environment(activate_env = true, instantiate = false)

# Create output directories
plots_dir = joinpath(project_root, "src", "Data", "NODE_HyperOpt_Strategies", "Plots_Results")
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
using LeastSquaresOptim  # For LM

# Global optimization
using Evolutionary
using BlackBoxOptim

# Hyperparameter optimization
using Hyperopt
using BayesianOptimization
using GaussianProcesses  # For Bayesian Optimization

# Utilities
using LinearAlgebra, Statistics, Random
using ComponentArrays, Lux, Zygote, StableRNGs
using DataFrames, CSV
using JSON3
using ProgressMeter

# -----------------------------------------------------------------------------
# Load your UDE setup (using your existing code)
println("Loading dataset and setting up UDE...")

using MATLAB
filename_DataSet = joinpath(project_root, "dataset", "LearnigData_Rod_Clamp_Pin_Rot_X02_Y00_mode2_train_085.mat")
mf_DataSet = MatFile(filename_DataSet)
data_DataSet = get_mvariable(mf_DataSet, "DataSet_train")
DataSet = jarray(data_DataSet)

# Use subset for hyperopt speed
num_trajectories = size(DataSet, 1)
trajectories_to_use = min(10, num_trajectories)  # Adjust based on computational budget

# Initialize variables and load data (abbreviated)
θ₀_true, κ₀_true = [], []
px_true, py_true, θ_true, κ_true = Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[]

ii = trajectories_to_use
for i in 1:ii
    θ₀_i  = Float32.(DataSet[i,3])
    κ₀_i  = Float32.(DataSet[i,112])
    push!(θ₀_true,  θ₀_i)
    push!(κ₀_true,  κ₀_i)
    
    pX_i = Float32.(DataSet[i,12:61])
    pY_i = Float32.(DataSet[i,62:111])
    θ_i  = Float32.(DataSet[i,162:211])
    κ_i  = Float32.(DataSet[i,112:161])
    push!(px_true, pX_i)
    push!(py_true, pY_i)
    push!(θ_true,  θ_i)
    push!(κ_true,  κ_i)
end

X_ssol_all = [hcat(px_true[i], py_true[i], θ_true[i], κ_true[i])' for i in 1:ii]

# Setup neural network
rng = StableRNG(1111)
m, n = 4, 3
layers = [m, 20, 20, 20, n]

const U = Lux.Chain(
    [Dense(fan_in => fan_out, Lux.tanh) for (fan_in, fan_out) in zip(layers[1:end-2], layers[2:end-1])]...,
    Dense(layers[end-1] => layers[end], identity),
)

p_init, st = Lux.setup(rng, U)
p_init = ComponentArray{Float32}(p_init)
const _st = st

# Define ODE and prediction functions
function ude_dynamics!(du, u, p, s)
    if any(!isfinite, u)
        du .= 0.0f0
        return nothing
    end
    û_1 = U(u, p, _st)[1]
    du[1] = û_1[1]
    du[2] = û_1[2]
    du[3] = u[4]
    du[4] = û_1[3]
    return nothing
end

sSpan = (0.0f0, 1.0f0)
s = Float32.(range(0, 1, length=50))
u0 = Float32.(X_ssol_all[1][:, 1])
const prob_nn = ODEProblem(ude_dynamics!, u0, sSpan, p_init)

function predict(θ, X_all = X_ssol_all, S = s)
    try
        [Array(solve(remake(prob_nn, u0 = Float32.(X[:, 1]), tspan = (S[1], S[end]), p = θ),
                AutoVern7(Rodas5()), saveat = S, abstol = 1e-6, reltol = 1e-6,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))) for X in X_all]
    catch e
        return nothing
    end
end

function loss(θ)
    X̂_sols = predict(θ)
    if X̂_sols === nothing
        return 1e10
    end
    total_loss = sum(mean(abs, X_ssol_all[i] .- X̂_sols[i]) for i in 1:length(X̂_sols))
    return total_loss / length(X̂_sols)
end

# -----------------------------------------------------------------------------
# STRATEGY IMPLEMENTATIONS

"""
    train_strategy_1(hyperparams; max_iters_dict)
    
Option 1: RAdam → LBFGS/LM
Hyperparameters:
- radam_lr: learning rate for RAdam
- radam_iters: iterations for RAdam  
- lbfgs_m: memory size for LBFGS
- use_lm: whether to use LM instead of LBFGS
"""
function train_strategy_1(hyperparams; max_iters_dict=Dict(:radam=>200, :refine=>500))
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
    
    # Phase 2: LBFGS or LM
    if hyperparams[:use_lm]
        println("  Strategy 1 - Phase 2: Levenberg-Marquardt")
        
        # Residual function for LM
        function residual_vector!(resid, θ)
            X̂_sols = predict(θ)
            if X̂_sols === nothing
                resid .= 1e10
                return resid
            end
            idx = 1
            for i in 1:length(X̂_sols)
                for j in 1:size(X_ssol_all[i], 1)
                    for k in 1:size(X_ssol_all[i], 2)
                        resid[idx] = X_ssol_all[i][j, k] - X̂_sols[i][j, k]
                        idx += 1
                    end
                end
            end
            return resid
        end
        
        total_residuals = sum(length(vec(X)) for X in X_ssol_all)
        
        t2 = @elapsed begin
            result = LeastSquaresOptim.optimize!(
                (resid, θ_vec) -> residual_vector!(resid, ComponentArray{Float32}(θ_vec, axes(p_current))),
                zeros(Float32, total_residuals),
                vec(p_current),
                LevenbergMarquardt();
                x_tol = hyperparams[:lm_xtol],
                f_tol = hyperparams[:lm_ftol],
                iterations = max_iters_dict[:refine],
                show_trace = false
            )
            p_current = ComponentArray{Float32}(result.minimizer, axes(p_current))
        end
    else
        println("  Strategy 1 - Phase 2: LBFGS")
        optprob_refine = Optimization.OptimizationProblem(optf, p_current)
        
        t2 = @elapsed begin
            res_lbfgs = Optimization.solve(optprob_refine,
                                          LBFGS(m = hyperparams[:lbfgs_m]),
                                          maxiters = max_iters_dict[:refine])
        end
        p_current = res_lbfgs.u
    end
    total_time += t2
    
    final_loss = loss(p_current)
    push!(all_losses, final_loss)
    
    return final_loss, p_current, all_losses, total_time
end

"""
    train_strategy_2(hyperparams; max_iters_dict)
    
Option 2: CMA-ES → LBFGS/LM
Hyperparameters:
- cmaes_popsize: population size (or auto)
- cmaes_iters: iterations for CMA-ES
- use_lm: whether to use LM instead of LBFGS
- lbfgs_m: memory size for LBFGS
"""
function train_strategy_2(hyperparams; max_iters_dict=Dict(:cmaes=>100, :refine=>500))
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
    
    # Phase 2: LBFGS or LM (same as Strategy 1)
    if hyperparams[:use_lm]
        println("  Strategy 2 - Phase 2: Levenberg-Marquardt")
        # [LM code same as Strategy 1]
        function residual_vector!(resid, θ)
            X̂_sols = predict(θ)
            if X̂_sols === nothing
                resid .= 1e10
                return resid
            end
            idx = 1
            for i in 1:length(X̂_sols)
                for j in 1:size(X_ssol_all[i], 1)
                    for k in 1:size(X_ssol_all[i], 2)
                        resid[idx] = X_ssol_all[i][j, k] - X̂_sols[i][j, k]
                        idx += 1
                    end
                end
            end
            return resid
        end
        
        total_residuals = sum(length(vec(X)) for X in X_ssol_all)
        
        t2 = @elapsed begin
            result = LeastSquaresOptim.optimize!(
                (resid, θ_vec) -> residual_vector!(resid, ComponentArray{Float32}(θ_vec, axes(p_current))),
                zeros(Float32, total_residuals),
                vec(p_current),
                LevenbergMarquardt();
                x_tol = hyperparams[:lm_xtol],
                f_tol = hyperparams[:lm_ftol],
                iterations = max_iters_dict[:refine],
                show_trace = false
            )
            p_current = ComponentArray{Float32}(result.minimizer, axes(p_current))
        end
    else
        println("  Strategy 2 - Phase 2: LBFGS")
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x, q) -> loss(x), adtype)
        optprob_refine = Optimization.OptimizationProblem(optf, p_current)
        
        t2 = @elapsed begin
            res_lbfgs = Optimization.solve(optprob_refine,
                                          LBFGS(m = hyperparams[:lbfgs_m]),
                                          maxiters = max_iters_dict[:refine])
            p_current = res_lbfgs.u
        end
    end
    total_time += t2
    
    final_loss = loss(p_current)
    push!(all_losses, final_loss)
    
    return final_loss, p_current, all_losses, total_time
end

"""
    train_strategy_3(hyperparams; max_iters_dict)
    
Option 3: DE/PSO → RAdam → LBFGS/LM
Hyperparameters:
- use_de: true for DE, false for PSO
- de_popsize: population size for DE
- de_f: mutation factor
- de_cr: crossover rate
- pso_particles: number of particles
- pso_w: inertia weight
- global_iters: iterations for DE/PSO
- radam_lr: learning rate for RAdam
- radam_iters: iterations for RAdam
- use_lm: whether to use LM
"""
function train_strategy_3(hyperparams; max_iters_dict=Dict(:global=>50, :radam=>200, :refine=>500))
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
    
    # Phase 3: LBFGS or LM (same as previous strategies)
    if hyperparams[:use_lm]
        println("  Strategy 3 - Phase 3: Levenberg-Marquardt")
        # [LM code same as before]
        function residual_vector!(resid, θ)
            X̂_sols = predict(θ)
            if X̂_sols === nothing
                resid .= 1e10
                return resid
            end
            idx = 1
            for i in 1:length(X̂_sols)
                for j in 1:size(X_ssol_all[i], 1)
                    for k in 1:size(X_ssol_all[i], 2)
                        resid[idx] = X_ssol_all[i][j, k] - X̂_sols[i][j, k]
                        idx += 1
                    end
                end
            end
            return resid
        end
        
        total_residuals = sum(length(vec(X)) for X in X_ssol_all)
        
        t3 = @elapsed begin
            result = LeastSquaresOptim.optimize!(
                (resid, θ_vec) -> residual_vector!(resid, ComponentArray{Float32}(θ_vec, axes(p_current))),
                zeros(Float32, total_residuals),
                vec(p_current),
                LevenbergMarquardt();
                x_tol = hyperparams[:lm_xtol],
                f_tol = hyperparams[:lm_ftol],
                iterations = max_iters_dict[:refine],
                show_trace = false
            )
            p_current = ComponentArray{Float32}(result.minimizer, axes(p_current))
        end
    else
        println("  Strategy 3 - Phase 3: LBFGS")
        optprob_refine = Optimization.OptimizationProblem(optf, p_current)
        
        t3 = @elapsed begin
            res_lbfgs = Optimization.solve(optprob_refine,
                                          LBFGS(m = hyperparams[:lbfgs_m]),
                                          maxiters = max_iters_dict[:refine])
            p_current = res_lbfgs.u
        end
    end
    total_time += t3
    
    final_loss = loss(p_current)
    push!(all_losses, final_loss)
    
    return final_loss, p_current, all_losses, total_time
end

# -----------------------------------------------------------------------------
# HYPERBAND IMPLEMENTATION

println("\n" * "="^60)
println("HYPERBAND OPTIMIZATION FOR ALL STRATEGIES")
println("="^60)

function hyperband_for_strategy(strategy_fn, hyperspace, strategy_name; 
                                max_resource=400, η=3)
    """
    Hyperband for a given strategy
    η: downsampling factor (typically 3)
    max_resource: maximum iterations to allocate
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
                    # Continuous parameter
                    config[param_name] = param_range[1] + (param_range[2] - param_range[1]) * rand()
                elseif param_range isa Tuple{Int, Int}
                    # Integer parameter
                    config[param_name] = rand(param_range[1]:param_range[2])
                elseif param_range isa Vector
                    # Categorical parameter
                    config[param_name] = rand(param_range)
                elseif param_range isa Tuple{String, String} && param_range[1] == "log"
                    # Log-scale parameter
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
            
            # Evaluate all configurations with r_i resources
            results = []
            for (idx, config) in enumerate(configs[1:min(n_i, length(configs))])
                # Adjust iterations based on resource allocation
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
            
            # Keep best n_i/η configurations
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

# Define hyperparameter spaces for each strategy
hyperspace_strategy1 = [
    :radam_lr => ("log", "1e-5:1e-1"),
    :radam_iters => (100, 500),
    :use_lm => [true, false],
    :lbfgs_m => (5, 20),
    :lm_xtol => ("log", "1e-10:1e-6"),
    :lm_ftol => ("log", "1e-10:1e-6")
]

hyperspace_strategy2 = [
    :cmaes_popsize => ["auto", 20, 50, 100],
    :cmaes_iters => (50, 200),
    :use_lm => [true, false],
    :lbfgs_m => (5, 20),
    :lm_xtol => ("log", "1e-10:1e-6"),
    :lm_ftol => ("log", "1e-10:1e-6")
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
    :use_lm => [true, false],
    :lbfgs_m => (5, 20),
    :lm_xtol => ("log", "1e-10:1e-6"),
    :lm_ftol => ("log", "1e-10:1e-6")
]

# Run Hyperband for each strategy
hyperband_results = Dict()

println("\n" * "="^30 * " STRATEGY 1 " * "="^30)
best_config1, best_loss1, results1 = hyperband_for_strategy(
    train_strategy_1, hyperspace_strategy1, "Strategy 1: RAdam → LBFGS/LM",
    max_resource=300, η=3
)
hyperband_results["Strategy1"] = (config=best_config1, loss=best_loss1)

println("\n" * "="^30 * " STRATEGY 2 " * "="^30)
best_config2, best_loss2, results2 = hyperband_for_strategy(
    train_strategy_2, hyperspace_strategy2, "Strategy 2: CMA-ES → LBFGS/LM",
    max_resource=250, η=3
)
hyperband_results["Strategy2"] = (config=best_config2, loss=best_loss2)

println("\n" * "="^30 * " STRATEGY 3 " * "="^30)
best_config3, best_loss3, results3 = hyperband_for_strategy(
    train_strategy_3, hyperspace_strategy3, "Strategy 3: DE/PSO → RAdam → LBFGS/LM",
    max_resource=400, η=3
)
hyperband_results["Strategy3"] = (config=best_config3, loss=best_loss3)

# -----------------------------------------------------------------------------
# BAYESIAN OPTIMIZATION IMPLEMENTATION

println("\n" * "="^60)
println("BAYESIAN OPTIMIZATION FOR ALL STRATEGIES")
println("="^60)

using GaussianProcesses
using Optim  # For acquisition function optimization

"""
Bayesian Optimization with Gaussian Process surrogate
"""
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
            
            # Convert continuous to integer if needed
            if param in [:radam_iters, :cmaes_iters, :global_iters, :lbfgs_m, 
                        :de_popsize, :pso_particles]
                config[param] = round(Int, config[param])
            end
            
            # Convert to boolean if needed
            if param == :use_lm || param == :use_de
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
        # Normalize observations for numerical stability
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
        
        # Find next point by optimizing acquisition function
        result = optimize(acquisition, lower_bounds, upper_bounds, 
                         lower_bounds .+ (upper_bounds .- lower_bounds) .* rand(n_dims),
                         Fminbox(LBFGS()), 
                         Optim.Options(iterations=100))
        x_next = Optim.minimizer(result)
        
        # Convert to hyperparameters and evaluate
        config = Dict{Symbol, Any}()
        for (j, param) in enumerate(param_names)
            if log_scale[j]
                config[param] = exp(x_next[j])
            else
                config[param] = x_next[j]
            end
            
            # Type conversions
            if param in [:radam_iters, :cmaes_iters, :global_iters, :lbfgs_m,
                        :de_popsize, :pso_particles]
                config[param] = round(Int, config[param])
            end
            if param == :use_lm || param == :use_de
                config[param] = config[param] > 0.5
            end
        end
        
        # Evaluate new configuration
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
        if param in [:radam_iters, :cmaes_iters, :global_iters, :lbfgs_m,
                    :de_popsize, :pso_particles]
            best_config[param] = round(Int, best_config[param])
        end
        if param == :use_lm || param == :use_de
            best_config[param] = best_config[param] > 0.5
        end
    end
    
    return best_config, y_observed[best_idx], X_observed, y_observed
end

# Define bounds for Bayesian Optimization
bo_bounds_strategy1 = Dict(
    :radam_lr => ("log", 1e-5, 1e-1),
    :radam_iters => ("linear", 100.0, 500.0),
    :use_lm => ("linear", 0.0, 1.0),
    :lbfgs_m => ("linear", 5.0, 20.0),
    :lm_xtol => ("log", 1e-10, 1e-6),
    :lm_ftol => ("log", 1e-10, 1e-6)
)

bo_bounds_strategy2 = Dict(
    :cmaes_popsize => ("linear", 20.0, 100.0),
    :cmaes_iters => ("linear", 50.0, 200.0),
    :use_lm => ("linear", 0.0, 1.0),
    :lbfgs_m => ("linear", 5.0, 20.0),
    :lm_xtol => ("log", 1e-10, 1e-6),
    :lm_ftol => ("log", 1e-10, 1e-6)
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
    :use_lm => ("linear", 0.0, 1.0),
    :lbfgs_m => ("linear", 5.0, 20.0)
)

# Run Bayesian Optimization
bo_results = Dict()

println("\n" * "="^30 * " BO STRATEGY 1 " * "="^30)
bo_config1, bo_loss1, _, _ = bayesian_optimization_strategy(
    train_strategy_1, bo_bounds_strategy1, 
    "Strategy 1: RAdam → LBFGS/LM",
    n_initial=5, n_iterations=15
)
bo_results["Strategy1"] = (config=bo_config1, loss=bo_loss1)

println("\n" * "="^30 * " BO STRATEGY 2 " * "="^30)  
bo_config2, bo_loss2, _, _ = bayesian_optimization_strategy(
    train_strategy_2, bo_bounds_strategy2,
    "Strategy 2: CMA-ES → LBFGS/LM", 
    n_initial=5, n_iterations=15
)
bo_results["Strategy2"] = (config=bo_config2, loss=bo_loss2)

println("\n" * "="^30 * " BO STRATEGY 3 " * "="^30)
bo_config3, bo_loss3, _, _ = bayesian_optimization_strategy(
    train_strategy_3, bo_bounds_strategy3,
    "Strategy 3: DE/PSO → RAdam → LBFGS/LM",
    n_initial=5, n_iterations=15  
)
bo_results["Strategy3"] = (config=bo_config3, loss=bo_loss3)

# -----------------------------------------------------------------------------
# FINAL COMPARISON AND RECOMMENDATIONS

println("\n" * "="^60)
println("FINAL RESULTS COMPARISON")
println("="^60)

# Create comparison table
comparison_df = DataFrame(
    Strategy = String[],
    Method = String[],
    BestLoss = Float64[],
    Config = String[]
)

for (strategy_name, result) in hyperband_results
    push!(comparison_df, (strategy_name, "Hyperband", result.loss, 
                          string(result.config)))
end

for (strategy_name, result) in bo_results
    push!(comparison_df, (strategy_name, "Bayesian Opt", result.loss,
                          string(result.config)))
end

sort!(comparison_df, :BestLoss)
println(comparison_df)

# Save best configurations
best_configs = Dict(
    "Hyperband" => hyperband_results,
    "BayesianOpt" => bo_results,
    "Comparison" => comparison_df
)

open(joinpath(plots_dir, "best_hyperparameters_strategies.json"), "w") do io
    JSON3.write(io, best_configs)
end

# Identify overall best
overall_best = comparison_df[1, :]
println("\n" * "="^60)
println("🏆 OVERALL BEST CONFIGURATION")
println("="^60)
println("Strategy: $(overall_best.Strategy)")
println("Method: $(overall_best.Method)")
println("Loss: $(overall_best.BestLoss)")
println("Configuration: $(overall_best.Config)")
println("="^60)

# Plot convergence comparison
pl_comparison = plot(title="Hyperparameter Optimization Results",
                    xlabel="Strategy",
                    ylabel="Best Loss",
                    yaxis=:log10,
                    legend=:topright)

strategies = ["Strategy1", "Strategy2", "Strategy3"]
hb_losses = [hyperband_results[s].loss for s in strategies]
bo_losses = [bo_results[s].loss for s in strategies]

bar!(pl_comparison, strategies, hb_losses, label="Hyperband", alpha=0.7)
bar!(pl_comparison, strategies, bo_losses, label="Bayesian Opt", alpha=0.7)

savefig(pl_comparison, joinpath(plots_dir, "hyperopt_strategies_comparison.pdf"))
display(pl_comparison)

println("\n✅ Hyperparameter optimization complete!")
println("Results saved to: $plots_dir")