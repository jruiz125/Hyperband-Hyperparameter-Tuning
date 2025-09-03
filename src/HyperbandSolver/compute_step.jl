"""
Compute the next step in Hyperband algorithm
"""

function compute_step!(state::HyperbandState, prob::OptimizationProblem, alg::Hyperband)
    # Return the current configuration to evaluate
    if state.eval_idx <= length(state.T)
        config = state.T[state.eval_idx]
        state.current_resource = floor(Int, state.r_i)
        return config
    else
        # This shouldn't happen if is_terminated is working correctly
        error("No more configurations to evaluate")
    end
end
