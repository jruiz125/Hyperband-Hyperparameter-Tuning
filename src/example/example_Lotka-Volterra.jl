using Optimization
using OptimizationOptimisers
using ComponentArrays
include("../src/HyperbandSolver.jl")
using .HyperbandSolver

# Example: Optimize hyperparameters for Lotka-Volterra UDE
function create_hyperband_optimization_problem()
    # Define the hyperparameter space
    hp_bounds = ComponentVector(
        learning_rate = (1e-4, 1e-2),
        n_hidden = (16, 64),
        n_layers = (2, 6),
        activation = (1, 3)  # 1=tanh, 2=relu, 3=rbf
    )
    
    # Objective function that trains a UDE with given hyperparameters
    function train_ude_with_config(config, resource)
        lr = config[1]
        n_hidden = floor(Int, config[2])
        n_layers = floor(Int, config[3])
        activation_id = floor(Int, config[4])
        
        # Simulate training (replace with actual UDE training)
        # Lower loss for good hyperparameters
        base_loss = abs(lr - 0.003) * 100 + abs(n_hidden - 32) * 0.01 + abs(n_layers - 4) * 0.1
        
        # Improvement with more resources
        improvement = exp(-resource / 20)
        
        return base_loss * improvement + 0.01 * randn()
    end
    
    # Configuration generator
    function get_random_ude_config()
        return [
            hp_bounds.learning_rate[1] + rand() * (hp_bounds.learning_rate[2] - hp_bounds.learning_rate[1]),
            hp_bounds.n_hidden[1] + rand() * (hp_bounds.n_hidden[2] - hp_bounds.n_hidden[1]),
            hp_bounds.n_layers[1] + rand() * (hp_bounds.n_layers[2] - hp_bounds.n_layers[1]),
            hp_bounds.activation[1] + rand() * (hp_bounds.activation[2] - hp_bounds.activation[1])
        ]
    end
    
    # Create optimization problem
    f = OptimizationFunction((x, p) -> train_ude_with_config(x, 81))
    prob = OptimizationProblem(f, get_random_ude_config())
    
    return prob, get_random_ude_config
end

# Run Hyperband
prob, config_gen = create_hyperband_optimization_problem()

# Create Hyperband solver
hyperband = Hyperband(
    R = 81,
    η = 3,
    get_random_config = config_gen,
    resource_name = :epochs
)

# Initialize state
state = HyperbandState(0, 0.0, 0, 0, [], [], 0, 0, 0.0, 0, 0.0, [], Inf, 0, [])

# Run optimization
println("Starting Hyperband optimization...")
Optimization.initialize!(state, prob, hyperband)

while !Optimization.is_terminated(state, prob, hyperband)
    step_result = Optimization.compute_step!(state, prob, hyperband)
    
    # Evaluate the configuration (simulate training)
    loss = prob.f(step_result.config, prob.p)
    result = (minimum = loss, config = step_result.config)
    
    Optimization.handle_result!(state, prob, hyperband, result)
end

# Get final result
result = Optimization.get_result(state, prob, hyperband)

println("\n" * "="^60)
println("FINAL HYPERBAND RESULTS")
println("="^60)
println("Best configuration found:")
println("  Learning rate: $(result.u[1])")
println("  Hidden units: $(floor(Int, result.u[2]))")
println("  Layers: $(floor(Int, result.u[3]))")
println("  Activation: $(floor(Int, result.u[4]))")
println("Best loss: $(result.objective)")