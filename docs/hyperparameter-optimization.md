# Hyperparameter Optimization for UDE Training

This document contains resources and methodologies for hyperparameter optimization in Universal Differential Equations (UDE) training.

## Overview

The ClampedPinnedRodUDE project implements sophisticated hyperparameter optimization strategies combining:
- **Hyperband Algorithm** for efficient resource allocation
- **Bayesian Optimization** for intelligent parameter search
- **Multi-stage training strategies** (RAdam → LSQR, CMA-ES → LSQR, DE/PSO → RAdam → LSQR)

## Key Implementation

The main optimization file: [`src/solvers/development/Optimization_copilot_Bayessian_LSQR.jl`](../src/solvers/development/Optimization_copilot_Bayessian_LSQR.jl)

## External Resources

### Hyperband Algorithm
- **Primary Reference**: [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://2020blogfor.github.io/posts/2020/04/hyperband/)
  - Reference paper of the creators of the Hyperband method for automatic hyperparameters optimization
  
- **General Reference**: [Utilizing the HyperBand Algorithm for Hyperparameter Optimization](https://2020blogfor.github.io/posts/2020/04/hyperband/)
  - Excellent blog post explaining Hyperband theory and implementation
  - Provides practical insights into successive halving and resource allocation
  - Contains visual examples and algorithm breakdown

### Related Academic Papers
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization (JMLR 2017)](https://jmlr.org/papers/v18/16-558.html)
- [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](https://arxiv.org/abs/1807.01774)

### Implementation Details

#### Multi-Stage Training Strategies

1. **Strategy 1: RAdam → LSQR**
   - Phase 1: RAdam optimizer for initial convergence
   - Phase 2: LSQR refinement for precision

2. **Strategy 2: CMA-ES → LSQR**
   - Phase 1: CMA-ES for global exploration
   - Phase 2: LSQR for local refinement

3. **Strategy 3: DE/PSO → RAdam → LSQR**
   - Phase 1: Differential Evolution or Particle Swarm Optimization
   - Phase 2: RAdam for gradient-based refinement
   - Phase 3: LSQR for final precision

#### Hyperparameter Spaces

Each strategy optimizes different parameter sets:
- Learning rates (log-scale)
- Iteration counts
- LSQR-specific parameters (tolerance, regularization)
- Population sizes for evolutionary algorithms

## Usage

```julia
# Load the optimization module
include("src/solvers/development/Optimization_copilot_Bayessian_LSQR.jl")

# Run Hyperband optimization for all strategies
# Results are automatically saved to timestamped directories
```

## Results

- Optimization results are saved to `src/Data/Hyper_optim/[timestamp]/`
- Best configurations are exported as JSON files
- Comparison plots are generated automatically

## Dependencies

The optimization requires specific packages in order:
```julia
# Core packages
using Hyperopt, BayesianOptimization, GaussianProcesses
using Distributions, Optim  # For statistical functions and LBFGS
using IterativeSolvers, FiniteDiff  # For LSQR implementation
using Evolutionary  # For global optimization algorithms
```

## Notes

- LSQR (Least Squares QR) is used for final refinement due to its stability with ill-conditioned problems
- Hyperband efficiently allocates computational resources across different hyperparameter configurations
- Bayesian optimization provides intelligent exploration of the parameter space
