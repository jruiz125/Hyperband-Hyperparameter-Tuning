#!/usr/bin/env julia

"""
Script to set up local development environment
This will ensure Julia uses the local project instead of the global package
"""

using Pkg

# Change to the local project directory
cd(@__DIR__)
println("Setting up local development environment...")
println("Current directory: ", pwd())

# Remove the global package to avoid conflicts
try
    println("Removing global ClampFixedRodSolver package...")
    Pkg.rm("ClampFixedRodSolver")
    println("✓ Global package removed")
catch e
    println("Note: Global package not found or already removed")
end

# Activate the local project
println("Activating local project...")
Pkg.activate(".")
println("✓ Local project activated")

# Install dependencies
println("Installing dependencies...")
Pkg.instantiate()
println("✓ Dependencies installed")

# Add the package in development mode (this makes Julia use the local source)
println("Adding package in development mode...")
Pkg.develop(PackageSpec(path="."))
println("✓ Package added in development mode")

println("\n🎉 Setup complete!")
println("Now you can use: using ClampFixedRodSolver")
println("And Julia will use your local development copy.")
