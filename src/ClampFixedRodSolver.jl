"""
    ClampFixedRodSolver

A comprehensive module for solving the inverse kinematics of clamped-pinned rods 
using elliptical integrals. This module provides both computational solvers and 
data preparation utilities for rod mechanics simulations.

# Main Features
- Elliptical rod solver using MATLAB integration
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
module ClampFixedRodSolver

# Core dependencies
using MATLAB
using MLUtils
using JLD2
using Dates
using Statistics
using StableRNGs

# Always include utility modules
include("utils/project_utils.jl")
include("utils/config.jl")

# Include solver modules
include("solvers/elliptical_rod_solver.jl")
include("solvers/clamp_fixed_rod_solver.jl")

# Include MATLAB-dependent script functions
include("scripts/solve_and_prepare_data.jl")

# Export project utilities
export find_project_root, 
       setup_project_environment

# Export configuration functions and types
export ClampedRodConfig,
       get_default_config,
       create_config,
       print_config,
       should_save_figures,
       get_figure_path,
       get_figure_save_options

# Export main solver functions
export elliptical_rod_solver,
       clamp_fixed_rod_solver,
       solve_and_prepare_data

# Documentation synchronization function
"""
    sync_docs(; force::Bool=false)

Synchronize the documentation file (docs_for_language_server.jl) with exported functions.

# Arguments
- `force::Bool=false`: If true, regenerates all documentation

# Examples
```julia
using ClampFixedRodSolver
ClampFixedRodSolver.sync_docs()  # Add missing function docs
ClampFixedRodSolver.sync_docs(force=true)  # Regenerate all docs
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