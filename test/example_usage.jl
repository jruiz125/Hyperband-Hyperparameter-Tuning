# ---------------------------------------------------------------------------
# Example usage of the test solver function
# 
# This script demonstrates how to use the simplified elliptical rod solver
# function for testing and development purposes.
# ---------------------------------------------------------------------------

# Add the parent directory to the path so we can include source files
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Include required modules
include("test_elliptical_rod_solver.jl")

println("=== Example: Using Simplified Elliptical Rod Solver ===\n")

# Example 1: Using default configuration
println("--- Example 1: Default Configuration ---")
try
    success, results = elliptical_rod_solver_no_comparison()
    
    if success
        println("✓ Solver completed successfully!")
        println("Number of solutions found: $(results["nsols"])")
        if results["nsols"] > 0
            println("Final tip position: ($(results["computed_px_final"]), $(results["computed_py_final"]))")
            println("Energy: $(results["computed_energy"])")
        end
    else
        println("✗ Solver failed")
    end
catch e
    println("Error: $e")
    println("Note: This requires MATLAB engine to be properly configured")
end

println("\n" * "-"^50)

# Example 2: Using custom configuration
println("--- Example 2: Custom Configuration ---")
try
    # Create custom configuration
    custom_config = create_config(
        xp = 0.3,      # Target x position
        yp = 0.1,      # Target y position  
        mode = 2.0     # Buckling mode
    )
    
    success, results = elliptical_rod_solver_no_comparison(custom_config)
    
    if success
        println("✓ Custom solver completed successfully!")
        println("Target position: ($(custom_config.xp), $(custom_config.yp))")
        if results["nsols"] > 0
            println("Computed tip position: ($(results["computed_px_final"]), $(results["computed_py_final"]))")
            println("Position error: ($(abs(results["computed_px_final"] - custom_config.xp)), $(abs(results["computed_py_final"] - custom_config.yp)))")
        end
    else
        println("✗ Custom solver failed")
    end
catch e
    println("Error: $e")
    println("Note: This requires MATLAB engine to be properly configured")
end

println("\n" * "="^60)
println("To run actual tests, use: julia test/runtests.jl")
println("="^60)
