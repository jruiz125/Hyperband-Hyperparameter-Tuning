"""
Handle the result of a configuration evaluation
"""

function handle_result!(state::HyperbandState, prob::OptimizationProblem, 
                        alg::Hyperband, result)
    # Extract loss from result
    loss = result.loss
    push!(state.L, loss)
    state.total_evaluations += 1
    
    # Update best if needed
    if loss < state.best_loss
        state.best_loss = loss
        state.best_config = copy(result.config)
        println("  New best: loss = $(Printf.@sprintf("%.6f", loss)) at evaluation $(state.total_evaluations)")
    end
    
    # Move to next configuration
    state.eval_idx += 1
    
    # Check if we've evaluated all configs in this round
    if state.eval_idx > length(state.T)
        println("  Round $(state.i): evaluated $(length(state.T)) configs with r=$(state.r_i) $(alg.resource_name)")
        println("    Losses: min=$(Printf.@sprintf("%.6f", minimum(state.L))), mean=$(Printf.@sprintf("%.6f", mean(state.L)))")
        
        if state.i < state.s
            # SuccessiveHalving: keep top k configurations
            k = floor(Int, state.n_i / alg.η)
            state.T = top_k(state.T, state.L, k)
            println("    Selected top $k configurations")
            
            # Move to next round
            state.i += 1
            state.n_i = floor(Int, state.n * alg.η^(-state.i))
            state.r_i = state.r * alg.η^state.i
            
            # Reset for next round
            state.L = Float64[]
            state.eval_idx = 1
            
            println("  Starting round $(state.i) with $(length(state.T)) configs, r=$(state.r_i) $(alg.resource_name)")
        else
            # Finished this bracket
            push!(state.bracket_history, (
                s = state.s,
                n_configs = state.n,
                rounds = state.i + 1,
                best_loss = minimum(state.L),
                total_evals = length(state.L)
            ))
            
            println("Completed bracket s=$(state.s)")
            
            # Move to next bracket
            state.s -= 1
            
            if state.s >= 0
                # Initialize next bracket
                state.i = 0
                η = float(alg.η)
                R = float(alg.R)
                
                state.n = ceil(Int, state.B * η^state.s / (R * (state.s + 1)))
                state.r = R * η^(-state.s)
                state.n_i = state.n
                state.r_i = state.r
                
                # Generate new configurations
                state.T = get_hyperparameter_configuration(state.n, state, alg)
                state.L = Float64[]
                state.eval_idx = 1
                
                println("\n" * "="^60)
                println("Starting bracket s=$(state.s)")
                println("="^60)
                println("  n=$(state.n) configurations, r=$(state.r) $(alg.resource_name)")
            end
        end
    end
    
    return nothing
end
