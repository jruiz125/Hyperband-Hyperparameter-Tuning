"""
    ClampedPinnedRodUDE

A comprehensive module for solving the inverse kinematics of clamped-pinned rods 
using Universal Differential Equations (UDE). This module extends the ClampedPinnedRodSolver
with UDE methodology for neural ODE networks.

# Main Features
- UDE implementation for clamped-pinned rod problems
- Neural ODE networks for inverse position problems  
- Configuration management for rod parameters
- Project utilities for environment setup
- Data preparation and visualization tools

# Submodules
- `utils/`: Configuration and project management utilities
- `solvers/`: Core computational algorithms

# Testing
- Run tests with: `julia test/runtests.jl`
- Test framework includes validation and consistency checks
"""
module ClampedPinnedRodUDE

# Core dependencies

using BenchmarkTools
using ComponentArrays
using CpuId
using JLD2
using LaTeXStrings
using LineSearches
using Lux
using MATLAB
using Optimisers
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using OrdinaryDiffEq
using Plots
using SciMLSensitivity
using StableRNGs
using Statistics
using Pkg

# Always include utility modules
include("utils/project_utils.jl")
include("utils/config.jl")

# Include solver modules


# Export project utilities


# Export configuration functions and types


# Export main solver functions


# Documentation synchronization function
"""
    sync_docs(; force::Bool=false)

Synchronize the documentation file (docs_for_language_server.jl) with exported functions.

# Arguments
- `force::Bool=false`: If true, regenerates all documentation

# Examples
```julia
using ClampedPinnedRodSolver
ClampedPinnedRodSolver.sync_docs()  # Add missing function docs
ClampedPinnedRodSolver.sync_docs(force=true)  # Regenerate all docs
```
"""
function sync_docs(; force::Bool=false)
    sync_file = joinpath(@__DIR__, "utils", "docs", "sync_docs.jl")
    if isfile(sync_file)
        include(sync_file)
        return sync_documentation(force=force, verbose=true)
    else
        @warn "sync_docs.jl not found. Please ensure it exists in src/utils/docs/."
        return false
    end
end

export sync_docs

end # module