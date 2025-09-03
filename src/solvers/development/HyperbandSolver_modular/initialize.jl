"""
Initialize the Hyperband state
"""

function initialize!(state::HyperbandState, prob::OptimizationProblem, alg::Hyperband)
    # Calculate initial n and r for first bracket
    η = float(alg.η)
    R = float(alg.R)
    
    state.n = ceil(Int, state.B * η^state.s / (R * (state.s + 1)))
    state.r = R * η^(-state.s)
    state.n_i = state.n
    state.r_i = state.r
    
    # Generate initial configurations
    state.T = get_hyperparameter_configuration(state.n, state, alg)
    state.L = Float64[]
    state.eval_idx = 1
    state.current_resource = floor(Int, state.r_i)
    
    println("HYPERBAND initialized: s_max=$(state.s_max), B=$(state.B), R=$(alg.R), η=$(alg.η)")
    println("Starting bracket s=$(state.s) with n=$(state.n) configs, r=$(state.r) $(alg.resource_name)")
    
    return nothing
end
