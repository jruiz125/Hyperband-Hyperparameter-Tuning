"""
Configuration management for ClampedPinnedRodUDE.

This module provides a simplified configuration system for Neural ODE training and prediction,
focusing only on essential parameters for visualization and reproducibility.

# Configuration Categories
- Random Seed: For reproducible training runs
- File Saving: Figure saving options and paths  
- Plot Settings: Basic plot dimensions and styling
- CairoMakie 3D Plot: 3D visualization parameters

# Usage Example
```julia
using ClampedPinnedRodUDE

# Use default configuration
config = get_default_config()

# Custom configuration
config = create_config(
    random_seed = 2222,
    figure_format = "png", 
    plot_width = 1600,
    azimuth_3d = -0.2π
)

# Print configuration
print_config(config)
```
"""

using Dates

"""
    ClampedRodConfig

Configuration struct for ClampedPinnedRodUDE neural ODE training and prediction.

# Fields

## Random Seed & Reproducibility
- `random_seed::Int`: Random seed for reproducible training

## File Saving
- `save_figures::Bool`: Enable/disable figure saving
- `use_timestamped_folders::Bool`: Create timestamped folders for organization
- `figures_base_path::String`: Base directory path for saving figures
- `figure_format::String`: Figure file format ("png", "pdf", "svg", "eps")
- `figure_dpi::Int`: Figure resolution in DPI
- `pdf_version::String`: PDF version for CairoMakie compatibility

## Plot Settings
- `plot_width::Int`: Plot width in pixels
- `plot_height::Int`: Plot height in pixels
- `plot_margins::Float64`: Plot margins in millimeters
- `markersize::Int`: Scatter plot marker size

## CairoMakie 3D Plot
- `azimuth_3d::Float64`: 3D plot viewing angle in radians
- `aspect_ratio_3d::Tuple{Int,Int,Int}`: 3D plot aspect ratio (width, height, depth)
- `markersize_3d::Int`: 3D scatter plot marker size
- `alpha_3d::Float64`: 3D plot transparency (0.0 to 1.0)
- `strokewidth_3d::Float64`: 3D marker stroke width
"""
struct ClampedRodConfig
    # Random Seed & Reproducibility
    random_seed::Int
    
    # File Saving
    save_figures::Bool
    use_timestamped_folders::Bool
    figures_base_path::String
    figure_format::String
    figure_dpi::Int
    pdf_version::String
    
    # Plot Settings
    plot_width::Int
    plot_height::Int
    plot_margins::Float64
    markersize::Int
    
    # CairoMakie 3D Plot
    azimuth_3d::Float64
    aspect_ratio_3d::Tuple{Int,Int,Int}
    markersize_3d::Int
    alpha_3d::Float64
    strokewidth_3d::Float64
end

"""
    get_default_config()

Get the default configuration for ClampedPinnedRodUDE neural ODE training and prediction.

# Returns
- `ClampedRodConfig`: Configuration object with standard parameter values

# Default Values
- Random seed: 1111
- Figure saving: enabled with timestamped folders
- Plot size: 1200x500 pixels
- 3D plot azimuth: -0.4π radians
- Figure format: PDF
"""
function get_default_config()
    return ClampedRodConfig(
        # Random Seed & Reproducibility
        1111,       # random_seed: Random seed for reproducible training
        
        # File Saving
        true,       # save_figures: Enable figure saving
        true,       # use_timestamped_folders: Create timestamped folders
        "src/Data", # figures_base_path: Base path for figures
        "pdf",      # figure_format: Figure file format
        300,        # figure_dpi: Figure resolution in DPI
        "1.4",      # pdf_version: PDF version for CairoMakie
        
        # Plot Settings
        1200,       # plot_width: Plot width in pixels
        500,        # plot_height: Plot height in pixels
        10.0,       # plot_margins: Plot margins in mm
        3,          # markersize: Scatter plot marker size
        
        # CairoMakie 3D Plot
        -0.4π,      # azimuth_3d: 3D plot viewing angle
        (1,1,1),    # aspect_ratio_3d: 3D plot aspect ratio
        6,          # markersize_3d: 3D scatter marker size
        1.0,        # alpha_3d: 3D plot transparency
        0.5         # strokewidth_3d: 3D marker stroke width
    )
end

"""
    create_config(; kwargs...)

Creates a configuration with custom parameters by overriding default values.

# Keyword Arguments
- `random_seed::Int = 1111`: Random seed for reproducible training
- `save_figures::Bool = true`: Enable/disable figure saving
- `use_timestamped_folders::Bool = true`: Create timestamped folders
- `figures_base_path::String = "src/Data"`: Base path for figures
- `figure_format::String = "pdf"`: Figure file format ("png", "pdf", "svg", "eps")
- `figure_dpi::Int = 300`: Figure resolution in DPI
- `pdf_version::String = "1.4"`: PDF version for CairoMakie
- `plot_width::Int = 1200`: Plot width in pixels
- `plot_height::Int = 500`: Plot height in pixels
- `plot_margins::Float64 = 10.0`: Plot margins in mm
- `markersize::Int = 3`: Scatter plot marker size
- `azimuth_3d::Float64 = -0.4π`: 3D plot viewing angle
- `aspect_ratio_3d::Tuple{Int,Int,Int} = (1,1,1)`: 3D plot aspect ratio
- `markersize_3d::Int = 6`: 3D scatter marker size
- `alpha_3d::Float64 = 1.0`: 3D plot transparency
- `strokewidth_3d::Float64 = 0.5`: 3D marker stroke width

# Returns
- `ClampedRodConfig`: Configuration struct with custom values

# Examples
```julia
# Custom random seed for different runs
config = create_config(random_seed = 2222)

# High-quality figures
config = create_config(figure_dpi = 600, figure_format = "pdf")

# Large plots for presentations
config = create_config(plot_width = 1600, plot_height = 800)

# Custom 3D plot settings
config = create_config(azimuth_3d = -0.2π, markersize_3d = 8)
```
"""
function create_config(; kwargs...)
    defaults = get_default_config()
    return ClampedRodConfig(
        get(kwargs, :random_seed, defaults.random_seed),
        get(kwargs, :save_figures, defaults.save_figures),
        get(kwargs, :use_timestamped_folders, defaults.use_timestamped_folders),
        get(kwargs, :figures_base_path, defaults.figures_base_path),
        get(kwargs, :figure_format, defaults.figure_format),
        get(kwargs, :figure_dpi, defaults.figure_dpi),
        get(kwargs, :pdf_version, defaults.pdf_version),
        get(kwargs, :plot_width, defaults.plot_width),
        get(kwargs, :plot_height, defaults.plot_height),
        get(kwargs, :plot_margins, defaults.plot_margins),
        get(kwargs, :markersize, defaults.markersize),
        get(kwargs, :azimuth_3d, defaults.azimuth_3d),
        get(kwargs, :aspect_ratio_3d, defaults.aspect_ratio_3d),
        get(kwargs, :markersize_3d, defaults.markersize_3d),
        get(kwargs, :alpha_3d, defaults.alpha_3d),
        get(kwargs, :strokewidth_3d, defaults.strokewidth_3d)
    )
end

"""
    print_config(config::ClampedRodConfig)

Print configuration parameters for visualization and reproducibility settings.
"""
function print_config(config::ClampedRodConfig)
    println("ClampedRodConfig:")
    println("  Random Seed:")
    println("    random_seed: $(config.random_seed)")
    println("  File Saving:")
    println("    save_figures: $(config.save_figures)")
    println("    use_timestamped_folders: $(config.use_timestamped_folders)")
    println("    figures_base_path: \"$(config.figures_base_path)\"")
    println("    figure_format: \"$(config.figure_format)\"")
    println("    figure_dpi: $(config.figure_dpi)")
    println("    pdf_version: \"$(config.pdf_version)\"")
    println("  Plot Settings:")
    println("    plot_width: $(config.plot_width)")
    println("    plot_height: $(config.plot_height)")
    println("    plot_margins: $(config.plot_margins)")
    println("    markersize: $(config.markersize)")
    println("  CairoMakie 3D Plot:")
    println("    azimuth_3d: $(config.azimuth_3d)")
    println("    aspect_ratio_3d: $(config.aspect_ratio_3d)")
    println("    markersize_3d: $(config.markersize_3d)")
    println("    alpha_3d: $(config.alpha_3d)")
    println("    strokewidth_3d: $(config.strokewidth_3d)")
end

"""
    should_save_figures(config::ClampedRodConfig)

Check if figures should be saved based on configuration.
"""
function should_save_figures(config::ClampedRodConfig)
    return config.save_figures
end

"""
    get_figure_path(config::ClampedRodConfig, filename::String)

Get the full path for saving a figure based on configuration.
"""
function get_figure_path(config::ClampedRodConfig, filename::String)
    base_path = config.figures_base_path
    
    if config.use_timestamped_folders
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
        figure_dir = joinpath(base_path, timestamp)
        mkpath(figure_dir)
        return joinpath(figure_dir, filename)
    else
        mkpath(base_path)
        return joinpath(base_path, filename)
    end
end

"""
    get_figure_save_options(config::ClampedRodConfig)

Get figure saving options based on configuration.
"""
function get_figure_save_options(config::ClampedRodConfig)
    return Dict(
        :dpi => config.figure_dpi,
        :format => config.figure_format
    )
end
