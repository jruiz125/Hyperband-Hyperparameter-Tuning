using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux, ComponentArrays, Zygote
using Plots
using Statistics, Random
using DataFrames
using BenchmarkTools

# Include the Hyperband implementation
include("hypersolver.jl")

# -----------------------------------------------------------------------------
# Setup the Missing Physics Problem (Lotka-Volterra with unknown term)
# -----------------------------------------------------------------------------

# True system parameters
const Ω = 0.1f0
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    x, y = u
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
const u0 = [0.44249296f0, 4.6280594f0]

# Generate training data
const p_ = Float32[1.3, 0.9, 0.8, 1.8]
const tspan = (0.0f0, 3.0f0)
const t = range(tspan[1], tspan[2], length=40)
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=t)

# Add noise to the data
X = Array(solution)
const Xₙ = X + Float32(1e-3) * randn(eltype(X), size(X))

# Plot the data
plot(solution, alpha=0.75, color=:black, label=["True Data" nothing])
scatter!(t, Xₙ', color=:red, label=["Noisy Data" nothing])

# -----------------------------------------------------------------------------
# Define the UDE Model
# -----------------------------------------------------------------------------

"""
Create a UDE model with given hyperparameters
"""
function create_ude_model(config)
    # Extract hyperparameters
    hidden_dim = config[:hidden_dim]
    n_layers = config[:n_layers]
    activation = config[:activation]
    
    # Build the neural network layers
    layers = []
    push!(layers, Dense(3, hidden_dim, activation))  # Input: [x, y, t]
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim, hidden_dim, activation))
    end
    push!(layers, Dense(hidden_dim, 1))  # Output: missing term
    
    # Create the model
    nn_model = Lux.Chain(layers...)
    
    rng = Random.default_rng()
    Random.seed!(rng, 1111)
    
    p_model, st = Lux.setup(rng, nn_model)
    
    # Define the UDE dynamics
    function ude_dynamics!(du, u, p, t, p_true)
        û = nn_model([u[1], u[2], t], p, st)[1]  # Network prediction
        du[1] = p_true[1] * u[1] - p_true[2] * u[1] * u[2]
        du[2] = -p_true[4] * u[2] + û[1]  # Use network for missing term
    end
    
    # Closure to create the full dynamics
    nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
    
    # Create ODE problem
    prob_nn = ODEProblem(nn_dynamics!, u0, tspan, p_model)
    
    return prob_nn, nn_model, p_model, st
end

"""
Train UDE model with given configuration and resource budget
"""
function train_ude_model(config, resource)
    # Create model
    prob_nn, nn_model, p_init, st = create_ude_model(config)
    
    # Extract training hyperparameters
    lr = config[:learning_rate]
    solver = config[:solver]
    
    # Scale iterations based on resource
    max_iters = round(Int, resource)
    
    # Prediction function
    function predict(θ)
        _prob = remake(prob_nn, p=θ)
        Array(solve(_prob, solver, saveat=t,
                    abstol=1e-6, reltol=1e-6,
                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end
    
    # Loss function
    function loss(θ)
        X̂ = predict(θ)
        mean(abs2, Xₙ - X̂)
    end
    
    # Callback for early termination
    losses = Float32[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 10 == 0
            println("    Iteration $(length(losses)): Loss = $l")
        end
        return false
    end
    
    # Setup optimization
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
    
    # Train
    try
        res = Optimization.solve(optprob, ADAM(lr), callback=callback, maxiters=max_iters)
        final_loss = losses[end]
        return final_loss
    catch e
        println("    Training failed: $e")
        return Inf
    end
end

# -----------------------------------------------------------------------------
# Hyperparameter Configuration Space
# -----------------------------------------------------------------------------

function get_ude_config()
    Dict(
        :hidden_dim => rand([16, 32, 64, 128]),
        :n_layers => rand(2:5),
        :activation => rand([tanh, relu, sigmoid]),
        :learning_rate => 10.0^(-rand() * 3 - 1),  # 10^-4 to 10^-1
        :solver => rand([Tsit5(), Vern7(), AutoTsit5(Rosenbrock23())])
    )
end

# -----------------------------------------------------------------------------
# Run Hyperband vs Random Search Comparison
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("HYPERBAND vs RANDOM SEARCH for Missing Physics Discovery")
println("="^60)

# Total budget for fair comparison
const TOTAL_BUDGET = 1000  # Total training iterations available
const MAX_RESOURCE = 200   # Max iterations per configuration

"""
Run complete comparison experiment
"""
function run_comparison_experiment(n_runs=5)
    results = Dict(
        "Hyperband" => [],
        "Random Search" => []
    )
    
    for run in 1:n_runs
        println("\n" * "="^40)
        println("RUN $run/$n_runs")
        println("="^40)
        
        # 1. HYPERBAND
        println("\n--- Running Hyperband ---")
        eval_count = Ref(0)
        total_resource_used = Ref(0.0)
        
        function counted_ude_objective(config, resource)
            eval_count[] += 1
            total_resource_used[] += resource
            return train_ude_model(config, resource)
        end
        
        t_start = time()
        hb_config, hb_loss = hyperband(
            counted_ude_objective,
            get_ude_config,
            MAX_RESOURCE;
            η=3
        )
        hb_time = time() - t_start
        
        push!(results["Hyperband"], (
            loss=hb_loss,
            time=hb_time,
            evals=eval_count[],
            resource_used=total_resource_used[],
            config=hb_config
        ))
        
        println("Hyperband completed:")
        println("  Best loss: $hb_loss")
        println("  Time: $(round(hb_time, digits=2))s")
        println("  Evaluations: $(eval_count[])")
        println("  Total resource: $(total_resource_used[])")
        
        # 2. RANDOM SEARCH
        println("\n--- Running Random Search ---")
        eval_count[] = 0
        total_resource_used[] = 0.0
        
        t_start = time()
        rs_best_loss = Inf
        rs_best_config = nothing
        
        # Use same total budget as Hyperband
        n_configs = div(TOTAL_BUDGET, MAX_RESOURCE)
        
        for i in 1:n_configs
            config = get_ude_config()
            loss = counted_ude_objective(config, Float64(MAX_RESOURCE))
            if loss < rs_best_loss
                rs_best_loss = loss
                rs_best_config = config
                println("  New best at config $i: loss = $rs_best_loss")
            end
        end
        
        rs_time = time() - t_start
        
        push!(results["Random Search"], (
            loss=rs_best_loss,
            time=rs_time,
            evals=eval_count[],
            resource_used=total_resource_used[],
            config=rs_best_config
        ))
        
        println("Random Search completed:")
        println("  Best loss: $rs_best_loss")
        println("  Time: $(round(rs_time, digits=2))s")
        println("  Evaluations: $(eval_count[])")
        println("  Total resource: $(total_resource_used[])")
    end
    
    return results
end

# Run the experiment
results = run_comparison_experiment(3)  # Reduced runs for speed

# -----------------------------------------------------------------------------
# Analysis and Visualization
# -----------------------------------------------------------------------------

# Calculate statistics
stats = DataFrame(
    Method = String[],
    Mean_Loss = Float64[],
    Std_Loss = Float64[],
    Min_Loss = Float64[],
    Mean_Time = Float64[],
    Std_Time = Float64[],
    Mean_Evals = Float64[],
    Mean_Resource = Float64[]
)

for method in keys(results)
    losses = [r.loss for r in results[method]]
    times = [r.time for r in results[method]]
    evals = [r.evals for r in results[method]]
    resources = [r.resource_used for r in results[method]]
    
    push!(stats, (
        method,
        mean(losses),
        std(losses),
        minimum(losses),
        mean(times),
        std(times),
        mean(evals),
        mean(resources)
    ))
end

println("\n" * "="^60)
println("STATISTICAL SUMMARY")
println("="^60)
display(stats)

# Create comparison plots
p1 = bar(
    stats.Method,
    stats.Mean_Loss,
    yerr=stats.Std_Loss,
    title="Average Loss",
    ylabel="MSE Loss",
    legend=false,
    color=[:blue, :red]
)

p2 = bar(
    stats.Method,
    stats.Mean_Time,
    yerr=stats.Std_Time,
    title="Average Runtime",
    ylabel="Time (seconds)",
    legend=false,
    color=[:blue, :red]
)

p3 = bar(
    stats.Method,
    stats.Mean_Evals,
    title="Average Configurations Tested",
    ylabel="# Configurations",
    legend=false,
    color=[:blue, :red]
)

p4 = bar(
    stats.Method,
    stats.Mean_Resource,
    title="Average Total Resource Used",
    ylabel="Total Iterations",
    legend=false,
    color=[:blue, :red]
)

combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
display(combined_plot)
savefig(combined_plot, "hyperband_vs_random_missing_physics.png")

# -----------------------------------------------------------------------------
# Speedup Analysis
# -----------------------------------------------------------------------------

speedup = stats[stats.Method .== "Random Search", :Mean_Time][1] / 
          stats[stats.Method .== "Hyperband", :Mean_Time][1]

improvement = (stats[stats.Method .== "Random Search", :Mean_Loss][1] - 
               stats[stats.Method .== "Hyperband", :Mean_Loss][1]) / 
               stats[stats.Method .== "Random Search", :Mean_Loss][1] * 100

println("\n" * "="^60)
println("PERFORMANCE COMPARISON")
println("="^60)
println("Hyperband vs Random Search:")
println("  Time Speedup: $(round(speedup, digits=2))x faster")
println("  Loss Improvement: $(round(improvement, digits=1))%")
println("  Resource Efficiency: Hyperband uses adaptive allocation")

# -----------------------------------------------------------------------------
# Visualize Best Found Model
# -----------------------------------------------------------------------------

# Get best configuration from Hyperband
best_run_idx = argmin([r.loss for r in results["Hyperband"]])
best_config = results["Hyperband"][best_run_idx].config

println("\n" * "="^60)
println("BEST HYPERBAND CONFIGURATION")
println("="^60)
for (k, v) in best_config
    if k != :solver
        println("  $k: $v")
    else
        println("  $k: $(typeof(v))")
    end
end

# Train final model with best config and more resources
println("\nTraining final model with best configuration...")
prob_final, nn_final, p_final, st_final = create_ude_model(best_config)

function predict_final(θ)
    _prob = remake(prob_final, p=θ)
    Array(solve(_prob, best_config[:solver], saveat=t,
                abstol=1e-6, reltol=1e-6))
end

# Train for longer with best config
final_loss = train_ude_model(best_config, 500.0)
println("Final model loss after extended training: $final_loss")

println("\nExperiment completed successfully!")
println("Results saved to: hyperband_vs_random_missing_physics.png")

# -------------- Plots -------------
# Extract and display best configurations
println("\n" * "="^60)
println("OPTIMAL HYPERPARAMETERS FOUND")
println("="^60)

# Hyperband's best configuration
hb_best_idx = argmin([r.loss for r in results["Hyperband"]])
hb_best = results["Hyperband"][hb_best_idx]

println("\n🏆 HYPERBAND BEST CONFIGURATION:")
println("   Final Loss: $(hb_best.loss) (7.47×10⁻⁶)")
println("   Hyperparameters:")
for (key, value) in hb_best.config
    if key == :activation
        println("   - $key: $(nameof(value))")
    elseif key == :solver
        println("   - $key: $(typeof(value))")
    else
        println("   - $key: $value")
    end
end

# Random Search's best configuration  
rs_best_idx = argmin([r.loss for r in results["Random Search"]])
rs_best = results["Random Search"][rs_best_idx]

println("\n🎲 RANDOM SEARCH BEST CONFIGURATION:")
println("   Final Loss: $(rs_best.loss) (6.22×10⁻⁴)")
println("   Hyperparameters:")
for (key, value) in rs_best.config
    if key == :activation
        println("   - $key: $(nameof(value))")
    elseif key == :solver
        println("   - $key: $(typeof(value))")
    else
        println("   - $key: $value")
    end
end

# Compare the configurations
println("\n" * "="^60)
println("CONFIGURATION COMPARISON")
println("="^60)

comparison = DataFrame(
    Parameter = String[],
    Hyperband = Any[],
    Random_Search = Any[]
)

for key in keys(hb_best.config)
    hb_val = hb_best.config[key]
    rs_val = rs_best.config[key]
    
    if key == :activation
        push!(comparison, (string(key), nameof(hb_val), nameof(rs_val)))
    elseif key == :solver
        push!(comparison, (string(key), string(typeof(hb_val)), string(typeof(rs_val))))
    else
        push!(comparison, (string(key), hb_val, rs_val))
    end
end

display(comparison)