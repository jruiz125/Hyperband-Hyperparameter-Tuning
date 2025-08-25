"""
    ClampedRodConfig

Configuration parameters for clamped-pinned rod simulation.

# Fields
## Rod Geometry
- `L::Float64`: Length of the rod [m]
- `N::Int`: Number of nodes in which the rod is discretized
- `EI::Float64`: Stiffness of the angular component of deformation

## Clamped End Conditions
- `x0::Float64`: X Coordinate of Clamped-end [m]
- `y0::Float64`: Y Coordinate of Clamped-end [m]
- `theta::Float64`: Orientation of Clamped-end with X axis [rad]

## Linear Guide Parameters
- `alpha::Float64`: Angle of the linear guide [rad] (0 when not present)
- `lambda::Float64`: Length of the linear guide [m] (0 when not present)

## Pinned End Conditions
- `xp::Float64`: X coordinate of end-tip [m]
- `yp::Float64`: Y coordinate of end-tip [m]

## Solution Parameters
- `mode::Float64`: Buckling Mode in Elliptic Integrals approach
- `sol_number::Int`: Solution number from initial rod data (default: 1)

## Rotation Parameters (for clamp rotation data generation)
- `rotation_angle::Float64`: Total rotation angle [degrees] (default: 360.0)
- `angular_steps::Int`: Number of angular steps (default: 72)
- `save_at_step::Int`: Step number to save data before potential termination (default: 72)

## Grid Discretization Parameters (for elliptical solver)
- `slope::Float64`: Controls kr density distribution, 0-1 (default: 0.1)
- `Nkr::Int`: Number of points for kr axis discretization (default: 150)
- `Npsi::Int`: Number of points for psi axis discretization (default: 300)
- `epsilon::Float64`: Numerical tolerance to avoid singularities (default: 1e-9)

## Training Parameters (for dataset splitting)
- `train_ratio::Float64`: Ratio of dataset used for training, 0-1 (default: 0.7)

## Figure Saving Parameters
- `save_figures::Bool`: Enable/disable figure saving (default: true)
- `use_timestamped_folders::Bool`: Create timestamped folders for figures (default: true)
- `figures_base_path::String`: Base path for saving figures (default: "figures")
- `figure_format::String`: Figure file format: "png", "pdf", "svg", "eps" (default: "png")
- `figure_dpi::Int`: Figure resolution in DPI (default: 300)
"""

# Required imports
using Dates
struct ClampedRodConfig
    # Rod geometry
    L::Float64
    N::Int
    EI::Float64
    
    # Clamped end conditions
    x0::Float64
    y0::Float64
    theta::Float64
    
    # Linear guide parameters
    alpha::Float64
    lambda::Float64
    
    # Pinned end conditions
    xp::Float64
    yp::Float64
    
    # Solution parameters
    mode::Float64
    sol_number::Int
    
    # Rotation parameters
    rotation_angle::Float64
    angular_steps::Int
    save_at_step::Int
    
    # Grid discretization parameters
    slope::Float64
    Nkr::Int
    Npsi::Int
    epsilon::Float64
    
    # Training parameters
    train_ratio::Float64
    
    # Figure saving parameters
    save_figures::Bool
    use_timestamped_folders::Bool
    figures_base_path::String
    figure_format::String
    figure_dpi::Int
end

"""
    get_default_config()

Returns default configuration for the clamped-pinned rod UDE training.

# Returns
- `ClampedRodConfig`: Configuration struct with default values
"""
function get_default_config()
    return ClampedRodConfig(
        # Rod geometry
        1.0,    # L: Length of the rod [m]
        50,     # N: Number of nodes
        1.0,    # EI: Stiffness
        
        # Clamped end conditions
        0.0,    # x0: X Coordinate of Clamped-end [m]
        0.0,    # y0: Y Coordinate of Clamped-end [m]
        0.0,    # theta: Orientation of Clamped-end with X axis [rad]
        
        # Linear guide parameters
        0.0,    # alpha: Angle of the linear guide [rad]
        0.0,    # lambda: Length of the linear guide [m]
        
        # Pinned end conditions
        0.2,    # xp: X coordinate of end-tip [m]
        0.0,    # yp: Y coordinate of end-tip [m]
        
        # Solution parameters
        2.0,    # mode: Buckling Mode in Elliptic Integrals approach
        1,      # sol_number: Solution number from initial rod data
        
        # Rotation parameters
        360.0,  # rotation_angle: Total rotation angle [degrees]
        72,     # angular_steps: Number of angular steps
        72,     # save_at_step: Step to save data before potential termination
        
        # Grid discretization parameters
        0.1,    # slope: Controls kr density distribution
        150,    # Nkr: Number of points for kr axis discretization
        300,    # Npsi: Number of points for psi axis discretization
        1e-9,   # epsilon: Numerical tolerance to avoid singularities
        
        # Training parameters
        0.7,    # train_ratio: Ratio of dataset used for training (70%)
        
        # Figure saving parameters
        true,       # save_figures: Enable figure saving
        true,       # use_timestamped_folders: Create timestamped folders
        "figures",  # figures_base_path: Base path for figures
        "png",      # figure_format: Figure file format
        300,        # figure_dpi: Figure resolution in DPI
    )
end

"""
    create_config(; kwargs...)

Creates a configuration with custom parameters by overriding default values.

# Keyword Arguments
- `L::Float64 = 1.0`: Rod length [m]
- `N::Int = 50`: Number of grid points  
- `EI::Float64 = 1.0`: Bending rigidity
- `x0::Float64 = 0.0`: X Coordinate of Clamped-end [m]
- `y0::Float64 = 0.0`: Y Coordinate of Clamped-end [m]
- `theta::Float64 = 0.0`: Orientation of Clamped-end with X axis [rad]
- `alpha::Float64 = 0.0`: Angle of the linear guide [rad]
- `lambda::Float64 = 0.0`: Length of the linear guide [m]
- `xp::Float64 = 0.1`: X coordinate of end-tip [m]
- `yp::Float64 = 0.0`: Y coordinate of end-tip [m]
- `mode::Float64 = 2.0`: Buckling Mode in Elliptic Integrals approach
- `sol_number::Int = 1`: Solution number from initial rod data
- `rotation_angle::Float64 = 360.0`: Total rotation angle [degrees]
- `angular_steps::Int = 72`: Number of angular steps
- `save_at_step::Int = 72`: Step to save intermediate data (computation continues)
- `slope::Float64 = 0.1`: Controls kr density distribution (0-1)
- `Nkr::Int = 150`: Number of points for kr axis discretization
- `Npsi::Int = 300`: Number of points for psi axis discretization
- `epsilon::Float64 = 1e-9`: Numerical tolerance to avoid singularities
- `train_ratio::Float64 = 0.7`: Ratio of dataset used for training (0-1)
- `save_figures::Bool = true`: Enable/disable figure saving
- `use_timestamped_folders::Bool = true`: Create timestamped folders for figures
- `figures_base_path::String = "figures"`: Base path for saving figures
- `figure_format::String = "png"`: Figure file format ("png", "pdf", "svg", "eps")
- `figure_dpi::Int = 300`: Figure resolution in DPI

# Returns
- `ClampedRodConfig`: Configuration struct with custom values

# Examples
```julia
# Custom rod parameters
config = create_config(L = 2.0, N = 25, EI = 0.5)

# Custom boundary conditions
config = create_config(theta = π/4, xp = 0.3, yp = 0.1)

# With linear guide
config = create_config(alpha = π/6, lambda = 0.2)

# Custom rotation parameters
config = create_config(rotation_angle = 180.0, angular_steps = 36, save_at_step = 30)

# High resolution rotation
config = create_config(angular_steps = 360, save_at_step = 350)  # 1° increments

# Custom grid discretization
config = create_config(slope = 0.05, Nkr = 200, Npsi = 400)

# High precision grid
config = create_config(Nkr = 300, Npsi = 600)  # Finer mesh

# Custom numerical tolerance
config = create_config(epsilon = 1e-12)  # Higher precision

# Figure saving options
config = create_config(save_figures = false)  # Disable figure saving
config = create_config(use_timestamped_folders = false)  # No timestamped folders
config = create_config(figures_base_path = "my_results/plots")  # Custom path
config = create_config(figure_format = "pdf", figure_dpi = 600)  # High-res PDF figures

# No figure saving (for batch processing)
config = create_config(save_figures = false)

# Custom figure organization
config = create_config(
    save_figures = true,
    use_timestamped_folders = false, 
    figures_base_path = "results/rod_analysis",
    figure_format = "svg"
)
```
"""
function create_config(; 
    # Rod geometry
    L = 1.0, N = 50, EI = 1.0,
    
    # Clamped end conditions
    x0 = 0.0, y0 = 0.0, theta = 0.0,
    
    # Linear guide parameters
    alpha = 0.0, lambda = 0.0,
    
    # Pinned end conditions
    xp = 0.1, yp = 0.0,
    
    # Solution parameters
    mode = 2.0,
    sol_number = 1,
    
    # Rotation parameters (MATLAB 1-based indexing considerations)
        # rotation_angle = 360.0: Total rotation from 0° to 360° 
        # angular_steps = 72: Number of angular steps (Δθ = 360°/72 = 5° per step)
        # save_at_step = 73: Save iteration number (1-based indexing)
        #
        # INDEXING EXPLANATION:
        # - MATLAB uses 1-based indexing: iterations 1, 2, 3, ..., 73
        # - Julia uses 0-based arrays but MATLAB loop indices are 1-based
        # - For 72 angular steps with inclusive endpoints: iterations 1 to 73 (73 total)
        # - Angles: 0°, 5°, 10°, ..., 355°, 360° (0° and 360° are the same point)
        # - Trajectories: 73 total (including both 0° and 360° endpoints)
        #
        # EXPECTED BEHAVIOR:
        # - angular_steps=72 → 72 angular steps → 73 trajectories (inclusive)
        # - save_at_step=73 → save at iteration 73 (full rotation), continue to completion
        # - save_at_step=72 → save at iteration 72 (355°), continue to completion
    rotation_angle = 360.0, angular_steps = 72, save_at_step = angular_steps,
    
    # Grid discretization parameters
    slope = 0.1, Nkr = 150, Npsi = 300, epsilon = 1e-9,
    
    # Training parameters
    train_ratio = 0.7,
    
    # Figure saving parameters
    save_figures = true,
    use_timestamped_folders = true,
    figures_base_path = "figures",
    figure_format = "png",
    figure_dpi = 300
)
    return ClampedRodConfig(
        # Rod geometry
        L, N, EI,
        
        # Clamped end conditions
        x0, y0, theta,
        
        # Linear guide parameters
        alpha, lambda,
        
        # Pinned end conditions
        xp, yp,
        
        # Solution parameters
        mode,
        sol_number,
        
        # Rotation parameters
        rotation_angle, angular_steps, save_at_step,
        
        # Grid discretization parameters
        slope, Nkr, Npsi, epsilon,
        
        # Training parameters
        train_ratio,
        
        # Figure saving parameters
        save_figures, use_timestamped_folders, figures_base_path, figure_format, figure_dpi
    )
end

"""
    print_config(config::ClampedRodConfig)

Pretty-prints the configuration parameters in a readable format.

# Arguments
- `config::ClampedRodConfig`: Configuration to display
"""
function print_config(config::ClampedRodConfig)
    println("=== CLAMPED ROD CONFIGURATION ===")
    
    println("Rod Geometry:")
    println("  - Length (L): $(config.L) m")
    println("  - Grid Points (N): $(config.N)")
    println("  - Bending Rigidity (EI): $(config.EI)")
    
    println("Clamped End Conditions:")
    println("  - Position: ($(config.x0), $(config.y0)) m")
    println("  - Orientation (θ): $(config.theta) rad")
    
    println("Linear Guide Parameters:")
    println("  - Angle (α): $(config.alpha) rad")
    println("  - Length (λ): $(config.lambda) m")
    
    println("Pinned End Conditions:")
    println("  - Position: ($(config.xp), $(config.yp)) m")
    
    println("Solution Parameters:")
    println("  - Buckling Mode: $(config.mode)")
    println("  - Solution Number: $(config.sol_number)")
    
    println("Rotation Parameters:")
    println("  - Rotation Angle: $(config.rotation_angle) deg")
    println("  - Angular Steps: $(config.angular_steps)")
    println("  - Save at Step: $(config.save_at_step)")
    
    println("Grid Discretization:")
    println("  - Slope (kr density): $(config.slope)")
    println("  - Nkr (kr points): $(config.Nkr)")
    println("  - Npsi (psi points): $(config.Npsi)")
    println("  - Epsilon (tolerance): $(config.epsilon)")
    
    println("Training Parameters:")
    println("  - Train Ratio: $(config.train_ratio) ($(round(Int, config.train_ratio*100))% training, $(round(Int, (1-config.train_ratio)*100))% testing)")
    
    println("Figure Saving Parameters:")
    println("  - Save Figures: $(config.save_figures)")
    println("  - Use Timestamped Folders: $(config.use_timestamped_folders)")
    println("  - Base Path: $(config.figures_base_path)")
    println("  - Format: $(config.figure_format)")
    println("  - DPI: $(config.figure_dpi)")
    
    println("====================================\n")
    return nothing
end

"""
    get_figure_path(config::ClampedRodConfig, filename::String; create_dirs::Bool = true)

Generate the full path for saving figures based on configuration settings.

# Arguments
- `config::ClampedRodConfig`: Configuration containing figure saving parameters
- `filename::String`: Base filename for the figure (without extension)
- `create_dirs::Bool = true`: Whether to create directories if they don't exist

# Returns
- `String`: Full path for the figure file

# Examples
```julia
config = create_config(save_figures = true, use_timestamped_folders = true)
path = get_figure_path(config, "rod_analysis")
# Returns: "figures/2025-08-22_143022/rod_analysis.png"

config = create_config(save_figures = true, use_timestamped_folders = false)
path = get_figure_path(config, "rod_analysis") 
# Returns: "figures/rod_analysis.png"

config = create_config(save_figures = false)
path = get_figure_path(config, "rod_analysis")
# Returns: "" (empty string when figure saving is disabled)
```
"""
function get_figure_path(config::ClampedRodConfig, filename::String; create_dirs::Bool = true)
    # Return empty string if figure saving is disabled
    if !config.save_figures
        return ""
    end
    
    # Start with base path
    figure_dir = config.figures_base_path
    
    # Add timestamped folder if enabled
    if config.use_timestamped_folders
        timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
        figure_dir = joinpath(figure_dir, timestamp)
    end
    
    # Create directories if requested
    if create_dirs && !isdir(figure_dir)
        mkpath(figure_dir)
    end
    
    # Add file extension based on format
    if !endswith(filename, "." * config.figure_format)
        filename = filename * "." * config.figure_format
    end
    
    # Return full path
    return joinpath(figure_dir, filename)
end

"""
    should_save_figures(config::ClampedRodConfig)

Check if figures should be saved based on configuration.

# Arguments
- `config::ClampedRodConfig`: Configuration to check

# Returns
- `Bool`: true if figures should be saved, false otherwise

# Example
```julia
config = create_config(save_figures = true)
if should_save_figures(config)
    savefig(plot, get_figure_path(config, "my_plot"))
end
```
"""
function should_save_figures(config::ClampedRodConfig)
    return config.save_figures
end

"""
    get_figure_save_options(config::ClampedRodConfig)

Get figure saving options as a dictionary for use with plotting libraries.

# Arguments
- `config::ClampedRodConfig`: Configuration containing figure parameters

# Returns
- `Dict{Symbol, Any}`: Dictionary with figure saving options

# Example
```julia
config = create_config(figure_format = "pdf", figure_dpi = 600)
save_opts = get_figure_save_options(config)
# Returns: Dict(:dpi => 600, :format => "pdf")

# Usage with Plots.jl
savefig(plot, filepath; save_opts...)
```
"""
function get_figure_save_options(config::ClampedRodConfig)
    return Dict(
        :dpi => config.figure_dpi,
        :format => config.figure_format
    )
end