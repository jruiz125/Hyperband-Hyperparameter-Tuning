# Hyperband Hyperparameter Tuning for Universal Differential Equations

This repository implements the Hyperband algorithm for efficient hyperparameter optimization of Universal Differential Equations (UDEs). The implementation focuses on neural network-enhanced differential equations with applications to physics-informed machine learning.

## Overview

The Hyperband algorithm provides an efficient method for hyperparameter optimization by adaptively allocating resources to promising configurations. This implementation is specifically designed for UDE training, where hyperparameters include:

- Neural network architecture (layers, nodes, activation functions)
- Optimization parameters (learning rates, batch sizes, algorithms)
- Training schedules and resource allocation

## Features

- **Efficient Hyperparameter Search**: Implements the Hyperband algorithm with early stopping
- **UDE Integration**: Specialized for Universal Differential Equations with neural network components
- **Multiple Examples**: Harmonic oscillator and Lotka-Volterra equations
- **Comprehensive Analysis**: Performance metrics, convergence analysis, and visualization
- **Modular Design**: Reusable components for different ODE systems

## Project Structure

```
├── src/
│   ├── HyperbandSolver/         # Core Hyperband implementation
│   │   ├── HyperbandSolver.jl   # Main solver interface
│   │   ├── types.jl             # Data structures and types
│   │   ├── compute_step.jl      # Hyperband step computation
│   │   └── utils.jl             # Utility functions
│   ├── example/                 # Example applications
│   │   ├── example_Harmonic_Oscillator.jl
│   │   └── example_Lotka-Volterra.jl
│   └── test/                    # Test suite
├── docs/                        # Documentation
├── figures/                     # Generated plots and analysis
└── Project.toml                 # Julia package dependencies
```

## Examples

### Harmonic Oscillator

The harmonic oscillator example demonstrates UDE training for a simple physical system:

```julia
include("src/example/example_Harmonic_Oscillator.jl")
```

This example:
- Generates synthetic data with noise
- Defines a 10-dimensional hyperparameter space
- Runs Hyperband optimization
- Compares results with analytical solutions
- Provides comprehensive visualization

### Lotka-Volterra System

The predator-prey dynamics example shows UDE application to nonlinear systems:

```julia
include("src/example/example_Lotka-Volterra.jl")
```

## Key Components

### Hyperband Solver

The main solver implements the successive halving algorithm with:
- Adaptive resource allocation
- Early stopping mechanisms
- Configuration ranking and selection
- Comprehensive logging and monitoring

### UDE Training

Specialized training pipeline for Universal Differential Equations:
- Neural network integration with ODE solvers
- Physics-informed loss functions
- Multi-stage optimization (Adam + L-BFGS)
- Robust error handling and convergence checking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jruiz125/Hyperband-Hyperparameter-Tuning.git
cd Hyperband-Hyperparameter-Tuning
```

2. Install Julia dependencies:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

### Basic Usage

```julia
using HyperbandOptim

# Define hyperparameter space
hyperparams = define_hyperparameter_space()

# Run Hyperband optimization
result = hyperband_optimize(
    train_function,
    hyperparams,
    max_resource=100000,
    eta=3
)
```

### Custom Applications

To adapt for your ODE system:

1. Define your UDE dynamics
2. Specify hyperparameter ranges
3. Implement training and evaluation functions
4. Run Hyperband optimization

## Results

The implementation has been tested on:
- **Harmonic Oscillator**: Achieves <1% error vs analytical solution
- **Lotka-Volterra**: Successfully captures predator-prey dynamics
- **Performance**: 10-100x speedup vs exhaustive search

## Dependencies

- Julia 1.9+
- DifferentialEquations.jl
- Lux.jl
- Optimization.jl
- Plots.jl
- StatsBase.jl

## Contributing

Contributions are welcome! Please see the issues tab for current development needs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hyperband_ude_2024,
  title={Hyperband Hyperparameter Tuning for Universal Differential Equations},
  author={Jose Ruiz},
  year={2024},
  url={https://github.com/jruiz125/Hyperband-Hyperparameter-Tuning}
}
```

## References

- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, 18(1), 6765-6816.
- Rackauckas, C., et al. (2020). Universal differential equations for scientific machine learning. *arXiv preprint arXiv:2001.04385*.
