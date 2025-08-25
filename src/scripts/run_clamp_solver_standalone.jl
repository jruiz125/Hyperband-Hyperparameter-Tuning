#!/usr/bin/env julia

"""
Standalone Clamp Fixed Rod Solver Runner
==========================================

This script runs the clamp_fixed_rod_solver in isolation to prevent MATLAB crashes
from affecting the main pipeline. It's designed to be called as a separate process.

Usage:
    julia --project=. src/scripts/run_clamp_solver_standalone.jl

Features:
- Isolated MATLAB session 
- Enhanced crash recovery
- Independent execution environment
- Detailed logging and error reporting
"""

# Ensure we're in the right environment
using Pkg
Pkg.activate(".")

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load required modules
using Dates

# Load project modules
include("../utils/config.jl")
include("../utils/project_utils.jl")
include("../solvers/clamp_fixed_rod_solver.jl")

function main()
    println("🚀 STANDALONE CLAMP FIXED ROD SOLVER")
    println("=" ^ 50)
    println("Start time: $(Dates.now())")
    println("Process ID: $(getpid())")
    println("=" ^ 50)
    
    try
        # Setup environment
        println("🔧 Setting up project environment...")
        project_root = setup_project_environment(activate_env = true, instantiate = false)
        println("✓ Project root: $project_root")
        
        # Create configuration that matches the main pipeline
        println("⚙️ Creating solver configuration...")
        config = create_config(xp = 0.2, yp = 0.0)  # Match main pipeline config
        
        println("\n📋 CONFIGURATION:")
        println("  - xp: $(config.xp)")
        println("  - yp: $(config.yp)")
        println("  - mode: $(config.mode)")
        println("  - angular_steps: $(config.angular_steps)")
        println("  - save_at_step: $(config.save_at_step)")
        
        # Run the clamp solver
        println("\n🔄 Starting clamp fixed rod solver...")
        println("=" ^ 50)
        
        success = clamp_fixed_rod_solver(config)
        
        # Add a small delay to ensure any background file operations complete
        println("⏳ Ensuring all file operations complete...")
        sleep(2)
        
        println("\n" * "=" ^ 50)
        if success
            println("✅ CLAMP SOLVER COMPLETED SUCCESSFULLY!")
            println("✓ Learning data generated")
            println("✓ MATLAB session handled properly")
            exit(0)  # Success exit code
        else
            println("❌ CLAMP SOLVER FAILED")
            println("✗ Check logs for details")
            exit(1)  # Failure exit code
        end
        
    catch e
        println("\n❌ STANDALONE SOLVER CRASHED")
        println("Error: $e")
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        exit(2)  # Crash exit code
    finally
        println("\nEnd time: $(Dates.now())")
        println("Process finished.")
    end
end

# Run the main function
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
