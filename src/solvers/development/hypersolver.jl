"""
    hyperband(objective_fn, get_random_config, max_resource; η=3)
    
Correct implementation of Hyperband algorithm following Li et al. (2016)
Algorithm 1 from the paper.

# Arguments
- `objective_fn`: Function that takes (config, resource) and returns loss
- `get_random_config`: Function that returns a random configuration
- `max_resource`: Maximum resource (R in the paper)
- `η`: Downsampling rate (default 3)
"""
function hyperband(objective_fn, get_random_config, max_resource::Int; η::Int=3)
    R = float(max_resource)  # Convert to float for calculations
    η_float = float(η)
    s_max = floor(Int, log(R) / log(η_float))
    B = (s_max + 1) * R
    
    best_config = nothing
    best_loss = Inf
    
    println("HYPERBAND: s_max=$s_max, B=$B, R=$R, η=$η")
    
    # Outer loop: iterate over brackets
    for s in s_max:-1:0
        println("\n═══ Bracket s=$s ═══")
        
        # Initial number of configurations
        n = ceil(Int, B * η_float^s / (R * (s + 1)))
        
        # Initial resource per configuration  
        r = R * η_float^(-s)
        
        println("Initial: n=$n configurations, r=$r resources each")
        
        # Generate n random configurations (Line 3 in Algorithm 1)
        T = [get_random_config() for _ in 1:n]
        
        # Successive halving (Lines 4-9 in Algorithm 1)
        for i in 0:s
            # Calculate n_i and r_i (Lines 5-6)
            n_i = floor(Int, n * η_float^(-i))
            r_i = r * η_float^i
            
            println("  Round $i: evaluating $(length(T)) configs with r=$r_i")
            
            # Run and evaluate (Line 7)
            L = Float64[]
            for (idx, config) in enumerate(T)
                try
                    loss = objective_fn(config, r_i)
                    push!(L, loss)
                    
                    # Track best
                    if loss < best_loss
                        best_loss = loss
                        best_config = config
                        println("    ⭐ New best: loss=$best_loss")
                    end
                catch e
                    println("    Config $idx failed: $e")
                    push!(L, Inf)
                end
            end
            
            # Select top k configurations (Line 8)
            if i < s  # Don't select on last iteration
                k = floor(Int, n_i / η_float)
                sorted_indices = sortperm(L)
                T = T[sorted_indices[1:min(k, length(T))]]
                println("    Selected top $k configurations")
            end
        end
    end
    
    return best_config, best_loss
end

"""
    successivehalving(objective_fn, configs, budget; η=3)
    
Implements the SuccessiveHalving subroutine (inner loop of Hyperband)
"""
function successivehalving(objective_fn, configs, budget; η::Int=3)
    n = length(configs)
    η_float = float(η)
    r = budget / n  # Initial resource per config
    s = floor(Int, log(n) / log(η_float))
    
    T = copy(configs)
    
    for i in 0:s
        n_i = floor(Int, n * η_float^(-i))
        r_i = r * η_float^i
        
        # Evaluate all remaining configs
        L = [objective_fn(t, r_i) for t in T]
        
        # Keep top k
        if i < s
            k = floor(Int, n_i / η_float)
            sorted_indices = sortperm(L)
            T = T[sorted_indices[1:min(k, length(T))]]
        end
    end
    
    # Return best configuration
    final_losses = [objective_fn(t, budget) for t in T]
    best_idx = argmin(final_losses)
    
    return T[best_idx], final_losses[best_idx]
end
#=
# -------- Example ------
    # Example matching paper's neural network experiment
    using Random

    # Simulate training a neural network (as in the paper)
    function train_nn_simulation(config, resource)
        # Simulate loss that improves with more resources
        # and depends on hyperparameters
        lr = config[:learning_rate]
        batch_size = config[:batch_size]
        
        # Simulated loss (lower is better)
        base_loss = abs(lr - 0.01) + abs(batch_size - 32) / 100
        improvement = 1 - (1 - exp(-resource / 50))
        
        return base_loss * improvement + randn() * 0.01
    end

    # Configuration space (similar to paper)
    function get_nn_config()
        Dict(
            :learning_rate => 10.0^(-rand() * 4),  # 10^-4 to 10^0
            :batch_size => rand([16, 32, 64, 128]),
            :momentum => rand() * 0.5 + 0.5        # 0.5 to 1.0
        )
    end

    # Run Hyperband (as in paper with R=81, η=3)
    best_config, best_loss = hyperband(
        train_nn_simulation,
        get_nn_config,
        81;  # R = 81 as in the paper
        η=3  # η = 3 as in the paper
    )

    println("\nBest configuration found:")
    for (k, v) in best_config
        println("  $k: $v")
    end
    println("Best loss: $best_loss")

# ----------------------------------------------

    using Random
    using Statistics
    using Plots
    using DataFrames

    # Include the hyperband implementation
    include("example_main.jl")

    """
        run_hyperband_experiments()
        
    Run multiple experiments comparing Hyperband with random search and grid search
    """
    function run_hyperband_experiments(n_runs=10)
        results = Dict(
            "Hyperband" => [],
            "Random Search" => [],
            "Successive Halving" => []
        )
        
        Random.seed!(1234)
        
        for run in 1:n_runs
            println("\n" * "="^50)
            println("RUN $run/$n_runs")
            println("="^50)
            
            # Track function evaluations
            eval_count = Ref(0)
            
            # Modified objective to count evaluations
            function counted_objective(config, resource)
                eval_count[] += 1
                return train_nn_simulation(config, resource)
            end
            
            # 1. Hyperband
            eval_count[] = 0
            t_start = time()
            hb_config, hb_loss = hyperband(
                counted_objective,
                get_nn_config,
                81;
                η=3
            )
            hb_time = time() - t_start
            push!(results["Hyperband"], (
                loss=hb_loss,
                time=hb_time,
                evals=eval_count[],
                config=hb_config
            ))
            
            # 2. Random Search (with same budget)
            eval_count[] = 0
            t_start = time()
            rs_best_loss = Inf
            rs_best_config = nothing
            total_budget = 81 * 5  # Same as Hyperband's total budget
            n_configs = div(total_budget, 81)  # Use full resources for each
            
            for _ in 1:n_configs
                config = get_nn_config()
                loss = counted_objective(config, 81.0)
                if loss < rs_best_loss
                    rs_best_loss = loss
                    rs_best_config = config
                end
            end
            rs_time = time() - t_start
            push!(results["Random Search"], (
                loss=rs_best_loss,
                time=rs_time,
                evals=eval_count[],
                config=rs_best_config
            ))
            
            # 3. Successive Halving only (s=4 bracket)
            eval_count[] = 0
            t_start = time()
            n_sh = 81
            configs_sh = [get_nn_config() for _ in 1:n_sh]
            sh_config, sh_loss = successivehalving(
                counted_objective,
                configs_sh,
                81.0;
                η=3
            )
            sh_time = time() - t_start
            push!(results["Successive Halving"], (
                loss=sh_loss,
                time=sh_time,
                evals=eval_count[],
                config=sh_config
            ))
        end
        
        return results
    end

    """
        plot_results(results)
        
    Create comparison plots for the experiment results
    """
    function plot_results(results)
        # Extract data for plotting
        methods = collect(keys(results))
        
        # Calculate statistics
        stats = DataFrame(
            Method = String[],
            Mean_Loss = Float64[],
            Std_Loss = Float64[],
            Min_Loss = Float64[],
            Max_Loss = Float64[],
            Mean_Time = Float64[],
            Mean_Evals = Float64[]
        )
        
        for method in methods
            losses = [r.loss for r in results[method]]
            times = [r.time for r in results[method]]
            evals = [r.evals for r in results[method]]
            
            push!(stats, (
                method,
                mean(losses),
                std(losses),
                minimum(losses),
                maximum(losses),
                mean(times),
                mean(evals)
            ))
        end
        
        # Create plots
        p1 = bar(
            stats.Method,
            stats.Mean_Loss,
            yerr=stats.Std_Loss,
            title="Average Loss Comparison",
            ylabel="Loss",
            legend=false,
            color=[:blue, :red, :green]
        )
        
        p2 = bar(
            stats.Method,
            stats.Mean_Time,
            title="Average Runtime",
            ylabel="Time (seconds)",
            legend=false,
            color=[:blue, :red, :green]
        )
        
        p3 = bar(
            stats.Method,
            stats.Mean_Evals,
            title="Average Function Evaluations",
            ylabel="# Evaluations",
            legend=false,
            color=[:blue, :red, :green]
        )
        
        # Box plot for loss distribution
        all_losses = Dict(m => [r.loss for r in results[m]] for m in methods)
        p4 = boxplot(
            [all_losses[m] for m in methods],
            xticks=(1:length(methods), methods),
            title="Loss Distribution",
            ylabel="Loss",
            legend=false,
            color=[:blue, :red, :green]
        )
        
        combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
        
        return combined_plot, stats
    end

    # Run experiments
    println("Running Hyperband experiments...")
    results = run_hyperband_experiments(10)

    # Plot results
    plot_results, stats_table = plot_results(results)
    display(plot_results)

    println("\n" * "="^60)
    println("STATISTICAL SUMMARY")
    println("="^60)
    display(stats_table)

    # Best configurations found
    println("\n" * "="^60)
    println("BEST CONFIGURATIONS FOUND")
    println("="^60)

    for method in keys(results)
        best_run = argmin([r.loss for r in results[method]])
        best_config = results[method][best_run].config
        best_loss = results[method][best_run].loss
        
        println("\n$method:")
        println("  Best Loss: $best_loss")
        println("  Configuration:")
        for (k, v) in best_config
            println("    $k: $v")
        end
    end

    # Save results to file
    using JSON3
    open("hyperband_results.json", "w") do f
        JSON3.write(f, results)
    end

    println("\nResults saved to hyperband_results.json")
    =#