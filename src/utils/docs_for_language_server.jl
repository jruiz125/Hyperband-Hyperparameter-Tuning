"""
Documentation-friendly version of ClampFixedRodSolver functions for Language Server.
This file contains the same function signatures and docstrings without MATLAB dependencies.
Use this file for hover/IntelliSense documentation while developing.

✅ AUTOMATED SYNC: This file is automatically synchronized with exported functions using 
the documentation sync system in src/utils/docs/. Run the sync tools to update:
- From Julia: include("src/utils/docs/sync_docs.jl"); sync_documentation()
- From CLI: julia src/utils/docs/update_docs_simple.jl
- From package: ClampFixedRodSolver.sync_docs()

To test hover functionality, open this file and hover over function names.
"""

module ClampFixedRodSolver

using Pkg
using MLUtils, JLD2, Dates, Statistics, StableRNGs

# ==================================================================================
# PROJECT UTILITIES
# ==================================================================================

"""
    find_project_root(; start_dir = @__DIR__)

Auto-detect project root directory by looking for characteristic files.

# Arguments
- `start_dir::String = @__DIR__`: Starting directory for search

# Returns
- `String`: Path to the project root directory

# Throws
- `ErrorException`: If project root cannot be found
"""
function find_project_root end

"""
    setup_project_environment(; activate_env = true, instantiate = false)

Sets up the project environment for ClampFixedRodSolver.

This function automatically detects the project root directory and sets up the Julia 
environment for the ClampFixedRodSolver package. It searches for characteristic project 
files (like Project.toml and dataset/ directory) to locate the root.

# Keyword Arguments
- `activate_env::Bool = true`: Whether to activate the Julia environment using Pkg.activate()
- `instantiate::Bool = false`: Whether to run Pkg.instantiate() to install dependencies

# Returns
- `String`: Path to the project root directory

# Throws
- `ErrorException`: If project root cannot be found

# Examples
```julia
# Basic usage - activate environment in project root
project_root = setup_project_environment()

# With dependency installation
project_path = setup_project_environment(activate_env=true, instantiate=true)

# Just detect project root without activating
project_path = setup_project_environment(activate_env=false)
```

# Details
The function performs the following steps:
1. Searches parent directories for Project.toml and dataset/ folder
2. Changes working directory to project root
3. Optionally activates the Julia environment
4. Optionally installs dependencies via Pkg.instantiate()

This is typically the first function called when using ClampFixedRodSolver to ensure
the proper environment is set up.
"""
function setup_project_environment(; activate_env = true, instantiate = false)
    # Implementation in actual ClampFixedRodSolver module
end

"""
    find_project_root(; start_dir = @__DIR__)

Auto-detect project root directory by looking for characteristic files.

# Arguments
- `start_dir::String = @__DIR__`: Starting directory for search

# Returns
- `String`: Path to the project root directory

# Throws
- `ErrorException`: If project root cannot be found
"""
function find_project_root(; start_dir = @__DIR__)
    # Implementation in ClampFixedRodSolver module
end

"""
    create_config(; kwargs...)

Create a configuration object for the clamped rod solver.

# Keyword Arguments
- `xp::Float64`: X position of pinned end
- `yp::Float64`: Y position of pinned end  
- `mode::Int`: Solver mode
- `L::Float64`: Rod length
- `N::Int`: Number of discretization points
- `EI::Float64`: Flexural rigidity
- And many more configuration options...

# Returns
- `ClampedRodConfig`: Configuration object

# Examples
```julia
# Basic configuration
config = create_config(xp=0.3, yp=0.0, mode=2)

# Advanced configuration
config = create_config(
    xp=0.5, yp=0.2, mode=1,
    L=1.0, N=50, EI=1.0
)
```
"""
function create_config end

"""
    ClampedRodConfig

Configuration struct for the clamped rod solver containing all solver parameters,
figure settings, and output options.

# Fields
- `X_TARGET_INITIAL::Float64`: Initial X-coordinate target position (default: 2.0)
- `Y_TARGET_INITIAL::Float64`: Initial Y-coordinate target position (default: 0.0)
- `SOLVER_MODE::Int`: Solver mode selection (1 or 2, default: 2)
- `FIGURE_SAVE_FORMAT::String`: Format for saved figures ("png", "pdf", "svg", default: "png")
- `SAVE_FIGURES::Bool`: Whether to save generated figures (default: false)
- `FIGURE_DPI::Int`: Resolution for saved figures (default: 300)
- `OUTPUT_DIRECTORY::String`: Directory for output files (default: "figures")

# Constructor
```julia
config = ClampedRodConfig()  # Default values
config = ClampedRodConfig(X_TARGET_INITIAL=3.0, Y_TARGET_INITIAL=1.5)  # Custom values
```

# Examples
```julia
# Create configuration with defaults
config = ClampedRodConfig()

# Access and modify parameters
config.X_TARGET_INITIAL = 2.5
config.SAVE_FIGURES = true
config.FIGURE_SAVE_FORMAT = "pdf"
```

# Notes
- All fields are mutable and can be modified after creation
- Configuration validates parameter ranges where applicable
- Used by all solver functions to control behavior and output

# See Also
- [`create_config`](@ref): Function-based configuration creation
- [`get_default_config`](@ref): Get default configuration
- [`print_config`](@ref): Display configuration values
"""
mutable struct ClampedRodConfig end

"""
    get_default_config()

Get the default configuration for the clamped rod solver.

# Returns
- `ClampedRodConfig`: Default configuration object with standard parameter values.

# Examples
```julia
# Get default configuration
config = get_default_config()

# Use with solver
results = initial_rod_solver(config)

# Modify default configuration
config = get_default_config()
config.X_TARGET_INITIAL = 2.5
config.SAVE_FIGURES = true
```

# Notes
- Provides a convenient way to get standard configuration
- Equivalent to calling `ClampedRodConfig()` with default parameters
- Recommended starting point for most solver applications

# See Also
- [`ClampedRodConfig`](@ref): Configuration struct definition
- [`create_config`](@ref): Alternative configuration creation method
- [`print_config`](@ref): Display configuration values
"""
function get_default_config end

"""
    print_config(config::ClampedRodConfig)

Display configuration parameters in a formatted, readable way.

# Arguments
- `config::ClampedRodConfig`: Configuration object to display

# Example
```julia
config = get_default_config()
print_config(config)
```
"""
function print_config end

"""
    should_save_figures(config::ClampedRodConfig)

Check if figure saving is enabled in the configuration.

# Arguments
- `config::ClampedRodConfig`: Configuration object

# Returns
- `Bool`: `true` if figures should be saved, `false` otherwise
"""
function should_save_figures end

"""
    get_figure_path(config::ClampedRodConfig, filename::String; create_dirs::Bool = true)

Generate appropriate file path for saving figures based on configuration.

# Arguments
- `config::ClampedRodConfig`: Configuration object
- `filename::String`: Base filename for the figure

# Keyword Arguments
- `create_dirs::Bool = true`: Whether to create directories if they don't exist

# Returns
- `String`: Full path for saving the figure
"""
function get_figure_path end

"""
    get_figure_save_options(config::ClampedRodConfig)

Get figure saving options from configuration.

# Arguments
- `config::ClampedRodConfig`: Configuration object

# Returns
- Options for figure saving (format depends on implementation)
"""
function get_figure_save_options end
    # Implementation in ClampFixedRodSolver module
end

"""
    initial_rod_solver(config=nothing; mode=2, xp=0.3, yp=0.0)

Solve the inverse kinematics problem for a flexible rod using elliptical integrals.

# Arguments
- `config`: Configuration object (optional, uses defaults if not provided)

# Keyword Arguments
- `mode::Int = 2`: Solver mode
- `xp::Float64 = 0.3`: X position of pinned end
- `yp::Float64 = 0.0`: Y position of pinned end

# Returns
- `Dict`: Results containing rod shape, forces, and analysis data

# Examples
```julia
# Basic usage
results = initial_rod_solver()

# With custom parameters
results = initial_rod_solver(mode=1, xp=0.5, yp=0.2)

# With configuration object
config = create_config(xp=0.3, yp=0.0)
results = initial_rod_solver(config)
```
"""
function initial_rod_solver(config=nothing; mode=2, xp=0.3, yp=0.0)
    # Implementation in ClampFixedRodSolver module - interfaces with MATLAB
end

"""
    clamp_fixed_rod_solver(config::Union{ClampedRodConfig, Nothing} = nothing)

Generates learning data for clamped-pinned rod configurations by rotating the clamp through 360°.

This function interfaces with MATLAB to generate a dataset of rod configurations by rotating 
the clamped end through a complete 360° rotation while maintaining the pinned end at a fixed 
position. The solver uses an existing rod solution as initial configuration and incrementally 
rotates the clamp, solving inverse kinematics at each step to generate training data for 
machine learning applications.

# Arguments
- `config::Union{ClampedRodConfig, Nothing} = nothing`: Rod configuration parameters. 
  If `nothing`, uses default configuration from `get_default_config()`.

# Returns
- `Bool`: `true` if data generation completed successfully, `false` if errors occurred

# Algorithm Steps
1. **Load Initial Configuration**: Uses parametric rod data file as starting point
2. **Setup Rotation Parameters**: Defines 360° rotation divided into configurable steps
3. **Incremental Solving**: For each rotation angle, solve inverse kinematics
4. **Visualization**: Generate real-time plots showing rod motion and analysis
5. **Data Export**: Save complete dataset as learning data for ML training

# Examples
```julia
# Using default configuration
success = clamp_fixed_rod_solver()

# Using custom configuration
config = create_config(xp = 0.5, yp = 0.0, mode = 2.0)
success = clamp_fixed_rod_solver(config)

# For different rod geometries
config = create_config(L = 2.0, EI = 0.5, N = 100)
success = clamp_fixed_rod_solver(config)

# High resolution rotation
config = create_config(angular_steps = 360, save_at_step = 350)
success = clamp_fixed_rod_solver(config)
```

# Application
Primarily used for generating training datasets for:
- **Neural Networks**: Learning inverse kinematics mappings
- **Machine Learning**: Pattern recognition in flexible rod behavior
- **Control Systems**: Developing model-based controllers
- **Robotics**: Soft/continuum robot control algorithms
"""
function clamp_fixed_rod_solver end

"""
    solve_and_prepare_data(config::Union{ClampedRodConfig, Nothing} = nothing)

Comprehensive solver and data preparation pipeline for rod mechanics.

This function provides a complete solution workflow including rod solving, data preparation,
and optional visualization. It serves as a high-level interface that coordinates multiple
solver functions and data processing steps.

# Arguments
- `config::Union{ClampedRodConfig, Nothing} = nothing`: Rod configuration parameters.
  If `nothing`, uses default configuration.

# Returns
- `Bool`: `true` if pipeline completed successfully, `false` if errors occurred

# Examples
```julia
# Basic usage with default configuration
success = solve_and_prepare_data()

# With custom configuration
config = create_config(xp=0.3, yp=0.1, mode=2)
success = solve_and_prepare_data(config)
```
"""
function solve_and_prepare_data end
