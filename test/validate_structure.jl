# ---------------------------------------------------------------------------
# Quick validation script for test structure
# 
# This script validates the test setup without running MATLAB-dependent tests.
# Use this to check if the test framework is properly configured.
# ---------------------------------------------------------------------------

using Test

println("=== ClampFixedRodSolver Test Structure Validation ===\n")

# Check if we're in the correct directory
current_dir = pwd()
println("Current directory: $current_dir")

# Check if test files exist
test_files = [
    "test/runtests.jl",
    "test/test_initial_rod_solver.jl", 
    "test/test_utils.jl",
    "test/README.md"
]

println("\n--- Checking test files ---")
all_files_exist = true
for file in test_files
    if isfile(file)
        println("✓ $file exists")
    else
        println("✗ $file missing")
        all_files_exist = false
    end
end

# Check if source files exist
source_files = [
    "src/utils/config.jl",
    "src/utils/project_utils.jl",
    "src/solvers/initial_rod_solver.jl"
]

println("\n--- Checking source files ---")
for file in source_files
    if isfile(file)
        println("✓ $file exists")
    else
        println("✗ $file missing")
        all_files_exist = false
    end
end

# Check Project.toml
println("\n--- Checking Project.toml ---")
if isfile("Project.toml")
    println("✓ Project.toml exists")
    content = read("Project.toml", String)
    if occursin("Test", content)
        println("✓ Test dependency found in Project.toml")
    else
        println("⚠ Test dependency not found in Project.toml")
    end
else
    println("✗ Project.toml missing")
    all_files_exist = false
end

# Try to include test utilities (syntax check)
println("\n--- Checking test utilities syntax ---")
try
    include("test/test_utils.jl")
    println("✓ test_utils.jl syntax is valid")
    
    # Test some utility functions
    @test TEST_TOLERANCE == 1e-6
    @test STRICT_TOLERANCE == 1e-12
    println("✓ Test constants are properly defined")
    
catch e
    println("✗ Error in test_utils.jl: $e")
    all_files_exist = false
end

# Final result
println("\n" * "="^50)
if all_files_exist
    println("✅ Test structure validation PASSED")
    println("\nTo run tests:")
    println("1. Ensure MATLAB engine is configured")
    println("2. Run: julia test/runtests.jl")
    println("3. Or use: julia --project=. -e \"using Pkg; Pkg.test()\"")
else
    println("❌ Test structure validation FAILED")
    println("Please check the missing files above.")
end
println("="^50)
