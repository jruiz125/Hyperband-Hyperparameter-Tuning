"""
Utility functions for Hyperband
"""

"""
    get_hyperparameter_configuration(n, state, alg)

Generate n random configurations
"""
function get_hyperparameter_configuration(n::Int, state::HyperbandState, alg::Hyperband)
    configs = Vector{Float64}[]
    
    for _ in 1:n
        if !isnothing(alg.get_random_config)
            push!(configs, alg.get_random_config())
        elseif !isnothing(state.lb) && !isnothing(state.ub)
            # Sample within bounds
            config = state.lb .+ rand(state.dim) .* (state.ub .- state.lb)
            push!(configs, config)
        else
            # Standard normal sampling
            push!(configs, randn(state.dim))
        end
    end
    
    return configs
end

"""
    top_k(configs, losses, k)

Select top k configurations based on losses
"""
function top_k(configs::Vector{Vector{Float64}}, losses::Vector{Float64}, k::Int)
    @assert length(configs) == length(losses)
    k = min(k, length(configs))
    sorted_indices = sortperm(losses)
    return configs[sorted_indices[1:k]]
end

"""
    evaluate_config(config, resource, prob, alg)

Evaluate a configuration with given resource budget
"""
function evaluate_config(config::Vector{Float64}, resource::Int, 
                         prob::OptimizationProblem, alg::Hyperband)
    if !isnothing(alg.inner_optimizer)
        # Use inner optimizer with resource limit
        modified_prob = remake(prob, u0=config)
        inner_opt = alg.inner_optimizer()
        
        try
            res = Optimization.solve(modified_prob, inner_opt; maxiters=resource)
            return (config=config, loss=res.objective, u=res.u)
        catch e
            return (config=config, loss=Inf, u=config)
        end
    else
        # Direct function evaluation
        loss = prob.f(config, prob.p)
        return (config=config, loss=loss, u=config)
    end
end
