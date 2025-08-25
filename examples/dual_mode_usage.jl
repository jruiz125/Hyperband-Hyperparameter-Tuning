# ==================================================================================
# Dual-Mode Usage Demonstration
# ==================================================================================
# 
# This file demonstrates how to use the ClampFixedRodSolver both as:
# 1. A standalone script (running solver files directly)
# 2. As a module (importing the package)
#
# For function documentation and hover functionality, see: docs_for_language_server.jl
#
# Author: José Luis Ruiz-Erezuma
# Created: August 2025
# ==================================================================================

println("=== Dual-Mode Usage Demonstration ===\n")


# ==================================================================================
# Method 1: Using as a Module (Recommended for most users)
# ==================================================================================

println("--- Method 1: Using as Module ---")

try
    using ClampFixedRodSolver

    
    # Setup environment (hover should work on the module-qualified version)
    setup_project_environment(activate_env=true)
    
    # Create configuration
    config = create_config(
        xp = 0.3,
        yp = 0.1, 
        mode = 2.0
    )
    
    println("✅ Module loaded successfully")
    println("✅ Configuration created")
    
    # Available functions through module:
    println("Available functions:")
    println("  - elliptical_rod_solver(config)")
    println("  - clamp_fixed_rod_solver(config)")
    println("  - solve_and_prepare_data(config)")
    println("  - get_default_config()")
    println("  - create_config(...)")
    
catch e
    println("❌ Module method failed: $e")
end

# ==================================================================================
# Method 2: Using Standalone Scripts (For development and customization)
# ==================================================================================

println("\n--- Method 2: Using Standalone Scripts ---")

# Example of how to run scripts standalone:
println("""
To run scripts standalone, use:

# 1. Run individual solver:
julia --project=. src/solvers/elliptical_rod_solver.jl

# 2. Run clamp solver:
julia --project=. src/solvers/clamp_fixed_rod_solver.jl

# 3. Run complete pipeline:
julia --project=. src/scripts/solve_and_prepare_data.jl

The scripts will automatically:
- Load required utilities (config, project_utils)
- Setup the environment
- Make functions available for direct use
""")

# ==================================================================================
# Method 3: Hybrid Approach (Best for development)
# ==================================================================================

println("--- Method 3: Hybrid Approach ---")

try
    # Load the module for basic functionality
    using ClampFixedRodSolver
    
    # Create configuration using module
    config = create_config(xp = 0.4, yp = 0.15, mode = 2.0)
    
    # But also include individual scripts for customization
    println("You can also include specific solvers for customization:")
    println("include(\"src/solvers/elliptical_rod_solver.jl\")")
    println("include(\"src/solvers/clamp_fixed_rod_solver.jl\")")
    
    println("✅ Hybrid approach ready")
    
catch e
    println("❌ Hybrid method failed: $e")
end

# ==================================================================================
# Example Configuration Options
# ==================================================================================

println("\n--- Example Configurations ---")

try
    using ClampFixedRodSolver
    
    # Default configuration
    config1 = get_default_config()
    println("Default config: xp=$(config1.xp), yp=$(config1.yp), mode=$(config1.mode)")
    
    # Custom configuration with specific tip position
    config2 = create_config(
        xp = 0.5,           # X position [m]
        yp = 0.2,           # Y position [m]
        mode = 3.0,         # Buckling mode
        angular_steps = 36  # 10° increments for rotation
    )
    println("Custom config: xp=$(config2.xp), yp=$(config2.yp), mode=$(config2.mode)")
    
    # Configuration for high-precision analysis
    config3 = create_config(
        xp = 0.3,
        yp = 0.1,
        mode = 2.0,
        Nkr = 200,          # Higher kr discretization
        Npsi = 400,         # Higher psi discretization
        angular_steps = 72  # 5° increments
    )
    println("High-precision config: Nkr=$(config3.Nkr), Npsi=$(config3.Npsi)")
    
catch e
    println("❌ Configuration examples failed: $e")
end

# ==================================================================================
# Usage Summary
# ==================================================================================

println("\n=== Usage Summary ===")
println("""
✅ Module Mode: Best for normal usage
   - using ClampFixedRodSolver
   - All functions available as exports
   - Clean, simple interface

✅ Standalone Mode: Best for development/customization  
   - julia --project=. src/solvers/[solver_name].jl
   - Direct access to internal functions
   - Easy to modify and test

✅ Hybrid Mode: Best for advanced users
   - Combine module imports with direct includes
   - Maximum flexibility
   - Can customize specific parts while using module infrastructure
""")

println("\n=== Demonstration Complete ===")
