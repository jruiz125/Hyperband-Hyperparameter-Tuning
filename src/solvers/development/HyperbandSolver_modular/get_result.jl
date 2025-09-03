"""
Get the final result from Hyperband optimization
"""

function get_result(state::HyperbandState, prob::OptimizationProblem, alg::Hyperband)
    println("\n" * "="^60)
    println("HYPERBAND COMPLETE")
    println("="^60)
    println("Total evaluations: $(state.total_evaluations)")
    println("Best loss: $(Printf.@sprintf("%.6f", state.best_loss))")
    println("Bracket history:")
    for bracket in state.bracket_history
        println("  s=$(bracket.s): $(bracket.n_configs) initial configs, " *
                "$(bracket.rounds) rounds, best_loss=$(Printf.@sprintf("%.6f", bracket.best_loss))")
    end
    
    # Return an OptimizationResult
    return Optimization.OptimizationResult(
        u = state.best_config,
        objective = state.best_loss,
        original = state.best_config,
        retcode = Optimization.ReturnCode.Success,
        stats = Dict(
            :total_evaluations => state.total_evaluations,
            :brackets_completed => length(state.bracket_history),
            :s_max => state.s_max,
            :B => state.B
        )
    )
end
