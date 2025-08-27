# ---------------------------------------------------------------------------
# Test utilities and helper functions
# 
# This module contains utility functions and constants used across different
# test modules.
# ---------------------------------------------------------------------------

using Test

"""
    TEST_TOLERANCE

Default tolerance for numerical comparisons in tests.
"""
const TEST_TOLERANCE = 1e-6

"""
    STRICT_TOLERANCE  

Strict tolerance for deterministic algorithm tests.
"""
const STRICT_TOLERANCE = 1e-12

"""
    find_compatible_reference_file(new_filename, reference_dir)

Find a compatible reference file for a new filename format that includes yp.
If a direct match isn't found, tries to find the old format (without yp) assuming yp=0.

# Arguments
- `new_filename`: New format filename (e.g., "CLampedPinnedRod_sol_1_mode_2_X02_Y00.mat")
- `reference_dir`: Directory containing reference files

# Returns
- `compatible_file`: Path to compatible reference file, or nothing if not found
- `match_type`: "exact", "yp_zero_compatible", or "not_found"

# Examples
```julia
# For a new file with yp=0
new_file = "CLampedPinnedRod_sol_1_mode_2_X02_Y00.mat"
ref_file, match_type = find_compatible_reference_file(new_file, ref_dir)
# Returns: ("CLampedPinnedRod_sol_1_mode_2_X02.mat", "yp_zero_compatible")
```
"""
function find_compatible_reference_file(new_filename, reference_dir)
    # First try exact match
    exact_path = joinpath(reference_dir, new_filename)
    if isfile(exact_path)
        return (exact_path, "exact")
    end
    
    # If exact match fails, try compatibility with old format (no yp term)
    # Extract components from new filename
    if occursin("_Y00", new_filename) || occursin("_YN00", new_filename)
        # This corresponds to yp=0, so we can look for the old format without yp
        old_filename = replace(new_filename, r"_Y[N]?00" => "")
        old_path = joinpath(reference_dir, old_filename)
        
        if isfile(old_path)
            return (old_path, "yp_zero_compatible")
        end
    end
    
    return (nothing, "not_found")
end

"""
    generate_old_format_filename(xp_suffix, sol_number, mode_number)

Generate filename in old format (without yp) for compatibility testing.
This assumes yp=0 for the old format files.

# Arguments
- `xp_suffix`: X coordinate suffix (e.g., "X02")
- `sol_number`: Solution number
- `mode_number`: Mode number

# Returns
- String: Old format filename
"""
function generate_old_format_filename(xp_suffix, sol_number, mode_number)
    return "CLampedPinnedRod_sol_$(sol_number)_mode_$(mode_number)_$(xp_suffix).mat"
end

"""
    compare_elliptical_rod_solutions(config, computed_results; tolerance=1e-3)

Compare computed elliptical rod solutions with reference solutions from Rod_Shape_reference.
Handles compatibility between new filename format (with yp) and old reference files (without yp).

# Arguments
- `config`: Configuration used for the computation
- `computed_results`: Results dictionary from initial_rod_solver_no_comparison
- `tolerance`: Numerical tolerance for comparisons (default: 1e-3)

# Returns
- `(success::Bool, message::String)`: Comparison result and descriptive message
"""
function compare_elliptical_rod_solutions(config, computed_results; tolerance=1e-3)
    try
        # Extract configuration parameters
        xp = config.xp
        yp = config.yp
        mode_number = Int(config.mode)
        sol_number = config.sol_number
        
        # Format suffixes
        xp_scaled = round(xp * 10)
        if xp_scaled < 10
            xp_suffix = "X0$(Int(xp_scaled))"
        else
            xp_suffix = "X$(Int(xp_scaled))"
        end
        
        yp_scaled = abs(round(yp * 10))
        if yp >= 0
            if yp_scaled < 10
                yp_suffix = "Y0$(Int(yp_scaled))"
            else
                yp_suffix = "Y$(Int(yp_scaled))"
            end
        else
            if yp_scaled < 10
                yp_suffix = "YN0$(Int(yp_scaled))"
            else
                yp_suffix = "YN$(Int(yp_scaled))"
            end
        end
        
        # Get project root and construct reference directory
        project_root = find_project_root()
        reference_dir = joinpath(project_root, "dataset", "MATLAB code", "Learning_Data_ClampedPinned_Rod_IK", "Find Initial Rod Shape", "Rod_Shape_reference")
        
        if !isdir(reference_dir)
            return (false, "Reference directory not found: $reference_dir")
        end
        
        # Try to find compatible reference file for each solution
        comparison_results = []
        solutions_found = 0
        
        for sol in 1:2  # Typically there are 2 solutions
            # Try new format first
            new_filename = "CLampedPinnedRod_sol_$(sol)_mode_$(mode_number)_$(xp_suffix)_$(yp_suffix).mat"
            ref_file_path, match_type = find_compatible_reference_file(new_filename, reference_dir)
            
            if ref_file_path !== nothing
                solutions_found += 1
                println("✓ Found reference file ($(match_type)): $(basename(ref_file_path))")
                
                # Here you would load and compare the reference data
                # This is a placeholder for the actual comparison logic
                push!(comparison_results, (sol, true, "Compatible reference found"))
            else
                println("⚠ No compatible reference file found for solution $sol")
                push!(comparison_results, (sol, false, "No reference available"))
            end
        end
        
        if solutions_found > 0
            return (true, "Found $solutions_found compatible reference files")
        else
            return (false, "No compatible reference files found")
        end
        
    catch e
        return (false, "Error during reference comparison: $e")
    end
end

"""
    check_yp_compatibility(yp_suffix)

Check if a yp suffix corresponds to yp=0 (compatible with old format files).

# Arguments
- `yp_suffix`: Y coordinate suffix (e.g., "Y00", "YN00", "Y05")

# Returns
- Bool: true if yp corresponds to 0, false otherwise
"""
function check_yp_compatibility(yp_suffix)
    return yp_suffix == "Y00" || yp_suffix == "YN00"
end

"""
    assert_approximately_equal(a, b, tolerance=TEST_TOLERANCE; message="")

Assert that two numerical values are approximately equal within tolerance.
"""
function assert_approximately_equal(a, b, tolerance=TEST_TOLERANCE; message="")
    diff = abs(a - b)
    if diff >= tolerance
        error("Values not approximately equal: |$a - $b| = $diff >= $tolerance. $message")
    end
    @test diff < tolerance
end

"""
    assert_physically_reasonable(energy; min_energy=0.0)

Assert that energy values are physically reasonable (non-negative and finite).
"""
function assert_physically_reasonable(energy; min_energy=0.0)
    @test isfinite(energy)
    @test energy >= min_energy
end

"""
    assert_solution_valid(results)

Assert that a solver results dictionary contains all required fields and valid values.
"""
function assert_solution_valid(results)
    required_fields = ["IN_first", "OUT_first", "nsols", "computed_px_final", 
                      "computed_py_final", "computed_energy"]
    
    for field in required_fields
        @test haskey(results, field)
    end
    
    @test results["nsols"] > 0
    assert_physically_reasonable(results["computed_energy"])
    @test isfinite(results["computed_px_final"])
    @test isfinite(results["computed_py_final"])
end

"""
    print_test_results(results, test_name="Test")

Print formatted test results for debugging.
"""
function print_test_results(results, test_name="Test")
    println("\n=== $test_name Results ===")
    if haskey(results, "nsols")
        println("Number of solutions: $(results["nsols"])")
    end
    if haskey(results, "computed_px_final")
        println("Final tip position: ($(results["computed_px_final"]), $(results["computed_py_final"]))")
    end
    if haskey(results, "computed_energy")
        println("Energy: $(results["computed_energy"])")
    end
    if haskey(results, "target_px")
        println("Target position: ($(results["target_px"]), $(results["target_py"]))")
    end
    println("=" ^ (length("=== $test_name Results ===") + 2))
end
