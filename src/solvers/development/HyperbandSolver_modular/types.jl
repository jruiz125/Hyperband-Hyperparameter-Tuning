"""
Type definitions for Hyperband solver
"""

"""
    Hyperband

Hyperband optimizer following Li et al. (2016)
"""
struct Hyperband
    R::Int                              # Maximum resource
    η::Int                              # Elimination factor
    get_random_config::Union{Function, Nothing}
    resource_name::Symbol
    inner_optimizer::Union{Function, Nothing}  # Optional inner optimizer factory
    
    function Hyperband(; R=81, η=3, get_random_config=nothing, 
                       resource_name=:iterations, inner_optimizer=nothing)
        @assert R > 0 "R must be positive"
        @assert η > 1 "η must be greater than 1"
        new(R, η, get_random_config, resource_name, inner_optimizer)
    end
end

"""
    HyperbandState

Mutable state for Hyperband algorithm
"""
mutable struct HyperbandState
    # Algorithm parameters
    s_max::Int
    B::Float64
    
    # Current position
    s::Int                          # Current bracket
    i::Int                          # Current round in bracket
    
    # Configurations
    T::Vector{Vector{Float64}}      # Current configurations
    L::Vector{Float64}              # Losses for current configs
    eval_idx::Int                   # Index being evaluated
    
    # Resource allocation
    n::Int                          # Initial configs for bracket
    r::Float64                      # Initial resource per config
    n_i::Int                        # Configs for current round
    r_i::Float64                    # Resource for current round
    current_resource::Int           # Resource for current evaluation
    
    # Best found
    best_config::Vector{Float64}
    best_loss::Float64
    
    # Tracking
    total_evaluations::Int
    bracket_history::Vector{NamedTuple}
    
    # Problem dimensions
    dim::Int
    lb::Union{Vector{Float64}, Nothing}
    ub::Union{Vector{Float64}, Nothing}
    
    function HyperbandState(prob::OptimizationProblem, alg::Hyperband)
        R = float(alg.R)
        η = float(alg.η)
        s_max = floor(Int, log(R) / log(η))
        B = (s_max + 1) * R
        
        dim = length(prob.u0)
        
        # Extract bounds if available
        lb = nothing
        ub = nothing
        if hasfield(typeof(prob), :lb) && hasfield(typeof(prob), :ub)
            lb = prob.lb
            ub = prob.ub
        end
        
        new(s_max, B, s_max, 0, Vector{Float64}[], Float64[], 1,
            0, 0.0, 0, 0.0, 0,
            copy(prob.u0), Inf, 0, NamedTuple[],
            dim, lb, ub)
    end
end
