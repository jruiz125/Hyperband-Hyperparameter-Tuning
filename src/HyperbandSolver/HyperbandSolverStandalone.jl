module HyperbandSolver

# Minimal dependencies - only use built-in Julia packages
using Random
using Statistics
using Printf

# =============================================================================
# CORE TYPES
# =============================================================================

"""
Configuration parameters for the Hyperband algorithm.
"""
struct Hyperband
    max_resource::Int
    η::Int
    
    function Hyperband(max_resource::Int, η::Int=3)
        @assert max_resource > 0 "max_resource must be positive"
        @assert η > 1 "η must be greater than 1"
        new(max_resource, η)
    end
end

"""
State tracking for Hyperband optimization.
"""
mutable struct HyperbandState
    current_bracket::Int
    current_round::Int
    configurations::Vector{Any}
    losses::Vector{Float64}
    best_config::Any
    best_loss::Float64
    total_evaluations::Int
    
    function HyperbandState()
        new(0, 0, Any[], Float64[], nothing, Inf, 0)
    end
end

# =============================================================================
# STANDALONE HYPERBAND ALGORITHM
# =============================================================================

"""
Standalone Hyperband function for backward compatibility and direct usage.
Follows Algorithm 1 from Li et al. (2016).

# Arguments
- `objective_fn`: Function that takes (config, resource) and returns loss
- `get_random_config`: Function that returns a random configuration
- `max_resource::Int`: Maximum resource allocation per configuration
- `η::Int=3`: Reduction factor for successive halving

# Returns
- `(best_config, best_loss)`: The best configuration found and its loss
"""
function hyperband(objective_fn, get_random_config, max_resource::Int; η::Int=3)
    R = float(max_resource)
    η_float = float(η)
    s_max = floor(Int, log(R) / log(η_float))
    B = (s_max + 1) * R
    
    best_config = nothing
    best_loss = Inf
    total_evaluations = 0
    
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
                total_evaluations += 1
                
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
    println("Total evaluations: $total_evaluations")
    
    return best_config, best_loss
end

# =============================================================================
# CONVENIENCE WRAPPER FOR HYPERBAND STRUCT
# =============================================================================

"""
Run hyperband optimization using the Hyperband struct.

# Arguments
- `hb::Hyperband`: Hyperband configuration
- `objective_fn`: Function that takes (config, resource) and returns loss
- `get_random_config`: Function that returns a random configuration

# Returns
- `(best_config, best_loss)`: The best configuration found and its loss
"""
function optimize(hb::Hyperband, objective_fn, get_random_config)
    return hyperband(objective_fn, get_random_config, hb.max_resource, η=hb.η)
end

# =============================================================================
# EXPORTS
# =============================================================================

export Hyperband, HyperbandState, hyperband, optimize

end # module
