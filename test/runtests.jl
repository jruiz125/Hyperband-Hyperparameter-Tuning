# ---------------------------------------------------------------------------
# Test runner for ClampedPinnedRodSolver
# 
# This file runs all tests for the elliptical rod solver project.
# To run all tests, execute: julia test/runtests.jl
# ---------------------------------------------------------------------------

using Test
using Pkg

# Activate the project environment (the ClampedPinnedRodSolver directory)
include("../src/utils/project_utils.jl")
project_root = setup_project_environment(activate_env=true, instantiate=false)

# Include test utilities
include("test_utils.jl")

# Include the test modules
include("test_initial_rod_solver.jl")
include("test_clamp_fixed_rod_solver.jl")

# Run all test suites
@testset "ClampedPinnedRodSolver Tests" begin
    @testset "Elliptical Rod Solver" begin
        test_initial_rod_solver()
    end
    
    @testset "Clamp Fixed Rod Solver" begin
        # Note: These tests require MATLAB engine - will gracefully handle if unavailable
        test_clamp_fixed_rod_solver()
    end
    
    # Future test suites can be added here
    # @testset "Configuration Tests" begin
    #     include("test_config.jl")
    # end
    
    # @testset "Project Utils Tests" begin
    #     include("test_project_utils.jl")
    # end
    
    # @testset "Performance Tests" begin
    #     include("test_performance.jl")
    # end
end

println("\n✓ All tests completed!")
