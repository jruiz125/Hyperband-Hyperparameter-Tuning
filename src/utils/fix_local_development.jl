#!/usr/bin/env julia

"""
Script to set up local development environment
This will ensure Julia uses the local project instead of the global package
"""

using Pkg

# Include the project utilities
include("project_utils.jl")

println("Setting up local development environment...")

# Remove the global package to avoid conflicts
try
    println("Removing global ClampedPinnedRodUDE package...")
    Pkg.rm("ClampedPinnedRodUDE")
    println("✓ Global package removed")
catch e
    println("Note: Global package not found or already removed")
end

# Use the setup_project_environment function to activate and install dependencies
project_root = setup_project_environment(activate_env=true, instantiate=true)

# Add the package in development mode (this makes Julia use the local source)
println("Adding package in development mode...")
try
    Pkg.develop(PackageSpec(path=project_root)) # "development mode" in Julia doesn't always require Pkg.develop() - when you're working within the package's own project environment, you're automatically using the local source code.
    println("✓ Package added in development mode")
catch e
    if occursin("same name or UUID as the active project", string(e))
        println("✓ Already in development mode (project is active)")
    else
        println("⚠️ Error adding package in development mode: ", e)
        rethrow(e)
    end
end

println("\n🎉 Setup complete!")
println("Now you can use: using ClampedPinnedRodUDE")
println("And Julia will use your local development copy.")
