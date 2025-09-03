module HyperbandSolver

using Optimization
using Random
using Statistics
using ComponentArrays
using Printf

# Include all component files
include("types.jl")
include("utils.jl")
include("initialize.jl")
include("compute_step.jl")
include("handle_result.jl")
include("is_terminated.jl")
include("get_result.jl")

# Export main types and functions
export Hyperband, HyperbandState, optimize_with_hyperband, hyperband

"""
    optimize_with_hyperband(prob, alg::Hyperband; kwargs...)

Main entry point for Hyperband optimization that works with Optimization.jl
"""
function optimize_with_hyperband(prob::OptimizationProblem, alg::Hyperband; kwargs...)
    # Create state
    state = HyperbandState(prob, alg)
    
    # Initialize
    initialize!(state, prob, alg)
    
    # Main optimization loop
    while !is_terminated(state, prob, alg)
        # Compute next step
        config_to_eval = compute_step!(state, prob, alg)
        
        # Evaluate the configuration
        result = evaluate_config(config_to_eval, state.current_resource, prob, alg)
        
        # Handle the result
        handle_result!(state, prob, alg, result)
    end
    
    # Return final result
    return get_result(state, prob, alg)
end

"""
    hyperband(objective_fn, get_random_config, max_resource; η=3)

Standalone Hyperband function for backward compatibility.
Follows Algorithm 1 from Li et al. (2016).
"""
function hyperband(objective_fn, get_random_config, max_resource::Int; η::Int=3)
    R = float(max_resource)
    η_float = float(η)
    s_max = floor(Int, log(R) / log(η_float))
    B = (s_max + 1) * R
    
    best_config = nothing
    best_loss = Inf
    
    println("HYPERBAND: s_max=$s_max, B=$B, R=$R, η=$η")
    
    # Outer loop: iterate over brackets (s from s_max down to 0)
    for s in s_max:-1:0
        println("\n" * "="^60)
        println("BRACKET s=$s")
        println("="^60)
        
        # Calculate n and r for this bracket
        n = ceil(Int, B * η_float^s / (R * (s + 1)))
        r = R * η_float^(-s)
        
        println("Initial n=$n configurations, r=$r resource per config")
        
        # Generate n random configurations
        configs = [get_random_config() for _ in 1:n]
        
        # SuccessiveHalving inner loop
        for i in 0:s
            # Calculate n_i and r_i for this round
            n_i = floor(Int, n * η_float^(-i))
            r_i = r * η_float^i
            
            # Number of configs to keep
            k = floor(Int, n_i / η_float)
            
            println("\nRound $i: evaluating $n_i configs with r=$r_i resource each")
            
            # Evaluate all configurations with current resource
            losses = Float64[]
            for (idx, config) in enumerate(configs)
                loss = objective_fn(config, floor(Int, r_i))
                push!(losses, loss)
                
                # Update best if needed
                if loss < best_loss
                    best_loss = loss
                    best_config = config
                    println("  New best found: loss = $loss")
                end
                
                # Progress indicator
                if idx % max(1, n_i ÷ 10) == 0
                    println("  Evaluated $idx/$n_i configs...")
                end
            end
            
            # Keep only top k configurations for next round
            if i < s
                # Sort configurations by loss
                sorted_indices = sortperm(losses)
                configs = configs[sorted_indices[1:k]]
                println("  Selected top $k configurations (losses: $(minimum(losses)) to $(losses[sorted_indices[k]]))")
            else
                println("  Final round complete. Best loss in bracket: $(minimum(losses))")
            end
        end
    end
    
    println("\n" * "="^60)
    println("HYPERBAND COMPLETE")
    println("="^60)
    println("Best loss: $best_loss")
    
    return best_config, best_loss
end

end # module
