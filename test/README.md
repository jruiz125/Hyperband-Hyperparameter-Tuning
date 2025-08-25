# EllipticalRodSolver Tests

This directory contains comprehensive tests for the EllipticalRodSolver project.

## Structure

- `runtests.jl` - Main test runner that executes all test suites
- `test_elliptical_rod_solver.jl` - Tests for the main elliptical rod solver function
- `test_clamp_fixed_rod_solver.jl` - Tests for the clamp fixed rod solver function
- `test_utils.jl` - Utility functions and constants for testing
- Future test files will be added here as the project grows

## Running Tests

### Run All Tests
```julia
# From the project root directory
julia test/runtests.jl
```

### Run in Package Mode
```julia
# In Julia REPL, from project root
julia> ]
pkg> activate .
pkg> test
```

### Run Individual Test Files
```julia
# From the project root directory
julia -e "using Pkg; Pkg.activate(\".\")"
julia test/test_elliptical_rod_solver.jl
```

## Test Categories

### 1. Default Configuration Test
- Tests the solver with default parameters
- Validates solution structure and physical reasonableness
- Ensures energy values are non-negative and finite

### 2. Custom Configuration Test  
- Tests the solver with custom parameter configurations
- Validates that target positions match input parameters
- Tests parameter flexibility

### 3. Algorithm Consistency Test
- Tests deterministic behavior by running identical configurations twice
- Ensures the algorithm produces identical results for identical inputs
- Uses strict tolerance (1e-12) for comparison

### 4. Solver Configuration Validation
- Tests multiple different rod configurations
- Validates that different inputs produce reasonable different outputs
- Tests various combinations of xp, yp, and mode parameters

## Clamp Fixed Rod Solver Tests

### 1. Basic Functionality Test
- Tests that the clamp solver function is callable and returns boolean results
- Uses fast configuration (reduced trajectories) for quick validation
- Handles MATLAB engine availability gracefully

### 2. Configuration Tests
- Tests default and custom configuration creation for clamp solver
- Validates parameter assignment and config structure
- Tests xp suffix generation and mode number handling

### 3. Reference Comparison Tests  
- Tests reference data loading and comparison functionality
- Validates statistical comparison metrics (max, mean, RMS differences)
- Tests with standard reference configurations (X02, mode 2, 72 steps)
- Gracefully handles cases where reference files are not available

### 4. Error Handling Tests
- Tests graceful handling of invalid configurations  
- Validates error recovery and meaningful error messages
- Tests MATLAB engine error handling

## Test Features

### Simplified Solver Function
The tests use `elliptical_rod_solver_no_comparison()`, which is a simplified version of the main solver that:
- Removes the reference comparison section
- Saves test results to separate folders (`Rod_Shape_Test`)
- Returns computed results in a structured dictionary
- Focuses on core algorithm validation

### Test Utilities
- `TEST_TOLERANCE` (1e-6) - Standard numerical tolerance
- `STRICT_TOLERANCE` (1e-12) - For deterministic behavior tests
- Helper functions for assertions and result validation
- Formatted output for debugging

## Prerequisites

1. **MATLAB Engine**: The tests require MATLAB to be properly configured
2. **Julia Packages**: Test, MATLAB, Dates packages
3. **Project Environment**: Tests activate the parent project environment

## Expected Test Behavior

✅ **Successful Tests**: All tests should pass if MATLAB is properly configured
⚠️ **MATLAB Issues**: If MATLAB engine is not available, tests will fail with specific guidance
🔍 **Debugging**: Test results include detailed output for debugging purposes

## Adding New Tests

To add new test categories:

1. Create new test files following the naming pattern `test_*.jl`
2. Include them in `runtests.jl`
3. Use the utilities from `test_utils.jl` for consistent behavior
4. Follow the existing test structure and documentation

## Test Data

- Test runs create separate dataset folders (`Rod_Shape_Test`) to avoid conflicts
- Test figures are saved with `Test_` prefix
- Test data is isolated from production runs
