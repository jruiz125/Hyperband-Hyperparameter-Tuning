"""
Check if Hyperband optimization is complete
"""

function is_terminated(state::HyperbandState, prob::OptimizationProblem, alg::Hyperband)
    # Terminate when all brackets have been processed
    return state.s < 0
end
