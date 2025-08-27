# ---------------------------------------------------------------------------
# Complete Rod Solver Pipeline Script
# 
# Can be run standalone or used through the ClampedPinnedRodSolver module
# ---------------------------------------------------------------------------
# Set this to true to force execution when using VSCode
if !@isdefined(FORCE_STANDALONE_EXECUTION)
    const FORCE_STANDALONE_EXECUTION = true
end

using Dates

# Setup for standalone usage (when run directly, not when included in module)
if (abspath(PROGRAM_FILE) == abspath(@__FILE__) || FORCE_STANDALONE_EXECUTION) && !@isdefined(ClampedRodConfig)
    include("../utils/project_utils.jl")
    include("../utils/config.jl")
    
    # Setup project environment for standalone mode
    project_root = setup_project_environment(activate_env = true, instantiate = false)
    
    # Set pipeline flag BEFORE including solvers to prevent their standalone execution
    # This flag indicates we're running the complete pipeline, not individual solvers
    global __PIPELINE_ALREADY_LOADED__ = true
    
    # Include solver functions for standalone mode
    include("../solvers/initial_rod_solver.jl")
    include("../solvers/clamp_fixed_rod_solver.jl")
end

# Include additional utilities
include("dataset_splitter.jl")
include("../utils/logging.jl")

"""
    run_with_complete_capture_integrated(config, log_dir)

Run the complete pipeline with full REPL output capture integrated into solve_and_prepare_data.
This is called when capture_all_output=true.
"""
function run_with_complete_capture_integrated(config, log_dir)
    # Generate log filename
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    # Fix floating-point precision issues by rounding and scaling
    # Use same format as dataset_splitter (with zero padding for single digits)
    xp_scaled = abs(round(config.xp * 10))
    if config.xp >= 0
        if xp_scaled < 10
            xp_str = "0$(Int(xp_scaled))"
        else
            xp_str = "$(Int(xp_scaled))"
        end
    else
        if xp_scaled < 10
            xp_str = "neg0$(Int(xp_scaled))"
        else
            xp_str = "neg$(Int(xp_scaled))"
        end
    end
    
    yp_scaled = abs(round(config.yp * 10))
    if config.yp >= 0
        if yp_scaled < 10
            yp_str = "0$(Int(yp_scaled))"
        else
            yp_str = "$(Int(yp_scaled))"
        end
    else
        if yp_scaled < 10
            yp_str = "neg0$(Int(yp_scaled))"
        else
            yp_str = "neg$(Int(yp_scaled))"
        end
    end
    mode_str = replace(string(Int(config.mode)), "." => "")
    
    log_filename = "RodSolver_CompleteREPL_X$(xp_str)_Y$(yp_str)_mode$(mode_str)_$(timestamp).log"
    
    # Ensure log directory exists
    if !isdir(log_dir)
        mkpath(log_dir)
    end
    
    log_path = joinpath(log_dir, log_filename)
    
    println("📝 Complete REPL output capture enabled: $(log_path)")
    println("📝 Note: All console output will now be captured to log file")
    
    # Execute with complete capture
    result = open(log_path, "w") do log_file
        # Write header
        write(log_file, "="^80 * "\n")
        write(log_file, "ROD SOLVER PIPELINE LOG - COMPLETE REPL OUTPUT CAPTURE\n")
        write(log_file, "="^80 * "\n")
        write(log_file, "Log file: $(log_path)\n")
        write(log_file, "Start time: $(now())\n")
        write(log_file, "Julia version: $(VERSION)\n")
        write(log_file, "Working directory: $(pwd())\n")
        write(log_file, "Capture mode: COMPLETE REPL OUTPUT CAPTURE\n")
        write(log_file, "="^80 * "\n")
        write(log_file, "CONFIGURATION PARAMETERS:\n")
        write(log_file, "="^80 * "\n")
        
        # Write configuration
        buffer = IOBuffer()
        show(buffer, config)
        config_str = String(take!(buffer))
        write(log_file, config_str * "\n")
        
        write(log_file, "="^80 * "\n")
        write(log_file, "COMPLETE PIPELINE EXECUTION OUTPUT (ALL REPL OUTPUT):\n")
        write(log_file, "="^80 * "\n")
        flush(log_file)
        
        # Redirect all output to log file
        pipeline_result = redirect_stdout(log_file) do
            redirect_stderr(log_file) do
                execute_pipeline_steps(config)
            end
        end
        
        # Write footer
        write(log_file, "\n")
        write(log_file, "="^80 * "\n")
        write(log_file, "PIPELINE COMPLETED\n")
        write(log_file, "End time: $(now())\n")
        write(log_file, "="^80 * "\n")
        
        return pipeline_result
    end
    
    # Final status message
    println("📝 Log file closed - Complete REPL output captured: $(log_path)")
    
    return result
end

"""
    execute_pipeline_steps(config)

Execute the complete pipeline steps with output capture.
"""
function execute_pipeline_steps(config)
    println()
    println("="^60)
    println("COMPLETE ROD SOLVER PIPELINE")
    println("="^60)
    println("Configuration Parameters:")
    println("   $(config)")
    println()
    
    println("="^60)
    println("STEP 1: Generating initial rod shape...")
    println("="^60)
    println("🔄 Starting initial_rod_solver...")
    
    # Step 1: Generate initial rod shape
    try
        success1 = initial_rod_solver(config)
        
        if !success1
            println("✗ initial_rod_solver failed")
            println("✗ Failed to generate initial rod shape")
            return false
        end
        
        println("✅ Initial rod shape generated successfully")
        
    catch e
        println("✗ Error in initial_rod_solver: $e")
        return false
    end
    
    println()
    println("="^60)
    println("STEP 2: Generating rotation learning data...")
    println("="^60)
    println("🔄 Starting clamp_fixed_rod_solver...")
    
    # Step 2: Generate learning data
    try
        success2 = clamp_fixed_rod_solver(config)
        
        if !success2
            println("✗ clamp_fixed_rod_solver failed")
            println("✗ Failed to generate learning data")
            return false
        end
        
        println("✅ Learning data generated successfully")
        
    catch e
        println("✗ Error in clamp_fixed_rod_solver: $e")
        return false
    end
    
    println()
    println("="^60)
    println("STEP 3: Splitting dataset for training/testing...")
    println("="^60)
    println("🔄 Starting split_dataset_for_training...")
    
    # Step 3: Split dataset
    try
        success3 = split_dataset_for_training(config)
        
        if !success3
            println("✗ split_dataset_for_training failed")
            println("✗ Failed to split dataset")
            return false
        end
        
        println("✅ Dataset split successfully")
        
    catch e
        println("✗ Error in split_dataset_for_training: $e")
        return false
    end
    
    println()
    println("="^60)
    println("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    println("="^60)
    
    return true
end

# -----------------------------------------------------------------------------

"""
    solve_and_prepare_data(config::ClampedRodConfig; enable_logging=true, log_dir="logs", capture_all_output=false)

Complete pipeline to generate initial rod shape and learning data for a given configuration.

This function combines three steps:
1. Generates the initial rod shape using elliptical integrals (initial_rod_solver)
2. Creates rotational learning data using the generated shape (clamp_fixed_rod_solver)
3. Splits the dataset into training and testing sets (split_dataset_for_training)

All output is logged to a timestamped log file for record keeping.

# Arguments
- `config::ClampedRodConfig`: Rod configuration parameters
- `enable_logging::Bool`: Enable logging to file (default: true)
- `log_dir::String`: Directory for log files (default: "logs")
- `capture_all_output::Bool`: Capture all REPL output including MATLAB engine output (default: false)
  - If `false`: Uses selective logging (structured pipeline events only)
  - If `true`: Uses complete REPL capture (all console output) - Note: Console output during execution will be redirected to log file

# Returns
- `Bool`: `true` if all steps completed successfully, `false` otherwise

# Generated Log File
Log filename format: `RodSolver_X{X}_Y{Y}_mode{M}_{timestamp}.log`

# Logging Modes
- `capture_all_output=false`: Selective logging of pipeline events and results
- `capture_all_output=true`: Complete REPL capture including all stdout/stderr output

# Example
```julia
custom_config = create_config(xp = 0.9, yp = 0.0, mode = 2.0)
success = solve_and_prepare_data(custom_config, enable_logging=true, capture_all_output=true)
```
# References
Based on work from University of the Basque Country UPV/EHU (Oscar Altuzarra, 2021 & 2025)
"""
function solve_and_prepare_data(config; enable_logging=true, log_dir="logs", capture_all_output=true)
    # If complete capture is requested, use a different approach
    if capture_all_output && enable_logging
        return run_with_complete_capture_integrated(config, log_dir)
    end
    
    # Setup logging if enabled (selective mode)
    log_capture = nothing
    if enable_logging
        log_capture = setup_logging(config, log_dir=log_dir, capture_all_output=false)
    end
    
    # Helper functions for conditional logging
    function safe_log_println(args...)
        if log_capture !== nothing
            log_println(log_capture, args...)
        else
            println(args...)
        end
    end
    
    function safe_log_section(title; width=60)
        if log_capture !== nothing
            log_section(log_capture, title, width=width)
        else
            println("\n" * "="^width)
            println(title)
            println("="^width)
        end
    end
    
    safe_log_section("COMPLETE ROD SOLVER PIPELINE")
    
    # Print configuration
    safe_log_println("Configuration Parameters:")
    if log_capture !== nothing
        # For logging, capture the config output
        buffer = IOBuffer()
        show(buffer, config)
        config_lines = split(String(take!(buffer)), '\n')
        for line in config_lines
            safe_log_println("  ", line)
        end
    else
        print_config(config)
    end
    
    # Step 1: Generate initial rod shape
    safe_log_section("STEP 1: Generating initial rod shape...")
    
    try
        success1 = if log_capture !== nothing
            capture_function_output(log_capture, initial_rod_solver, config, func_name="initial_rod_solver")
        else
            initial_rod_solver(config)
        end
        
        if !success1
            safe_log_println("✗ Failed to generate initial rod shape")
            if log_capture !== nothing
                finalize_logging(log_capture)
            end
            return false
        end
        
        safe_log_println("✓ Initial rod shape generated successfully")
        
    catch e
        safe_log_println("✗ Error in initial_rod_solver: $e")
        if log_capture !== nothing
            finalize_logging(log_capture)
        end
        return false
    end
    
    # Step 2: Generate learning data
    safe_log_section("STEP 2: Generating rotation learning data...")
    
    try
        success2 = if log_capture !== nothing
            capture_function_output(log_capture, clamp_fixed_rod_solver, config, func_name="clamp_fixed_rod_solver")
        else
            clamp_fixed_rod_solver(config)
        end
        
        if !success2
            safe_log_println("✗ Failed to generate learning data")
            if log_capture !== nothing
                finalize_logging(log_capture)
            end
            return false
        end
        
        safe_log_println("✓ Learning data generated successfully")
        
    catch e
        safe_log_println("✗ Error in clamp_fixed_rod_solver: $e")
        if log_capture !== nothing
            finalize_logging(log_capture)
        end
        return false
    end
    
    # Step 3: Split dataset for training and testing
    safe_log_section("STEP 3: Splitting dataset for training/testing...")
    
    try
        success3 = if log_capture !== nothing
            capture_function_output(log_capture, split_dataset_for_training, config, func_name="split_dataset_for_training")
        else
            split_dataset_for_training(config)
        end
        
        if !success3
            safe_log_println("✗ Failed to split dataset")
            if log_capture !== nothing
                finalize_logging(log_capture)
            end
            return false
        end
        
        safe_log_println("✓ Dataset split successfully")
        
    catch e
        safe_log_println("✗ Error in split_dataset_for_training: $e")
        if log_capture !== nothing
            finalize_logging(log_capture)
        end
        return false
    end
    
    safe_log_section("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    
    # Display file information
    safe_log_println("📁 DATA FILES LOCATION:")
    safe_log_println("Working directory: $(pwd())")
    
    # Generate the same filenames as in dataset_splitter for display
    # Fix floating-point precision issues by rounding and scaling
    xp_str = string(round(Int, config.xp * 10))
    if config.xp < 0
        xp_str = "neg" * xp_str
    end
    yp_str = string(round(Int, config.yp * 10))
    if config.yp < 0
        yp_str = "neg" * yp_str
    end
    mode_str = replace(string(Int(config.mode)), "." => "")
    
    # Format train/test ratios for filenames 
    # Ensure consistent formatting: pad to 3 digits (e.g., 0.85 -> "085", 0.9 -> "090")
    train_ratio_str = lpad(string(round(Int, config.train_ratio * 100)), 3, '0')
    test_ratio_str = lpad(string(round(Int, (1.0 - config.train_ratio) * 100)), 3, '0')
    
    base_filename = "LearnigData_Rod_Clamp_Pin_Rot_X$(xp_str)_Y$(yp_str)_$(config.angular_steps)sols_mode$(mode_str)"
    
    # The main dataset filename (contains all trajectories, no train_ratio suffix)
    full_dataset_filename = "$(base_filename).mat"
    
    # Generate rod filename from elliptical solver
    sol_number = config.sol_number
    rod_mode = Int(config.mode)
    rod_filename = "CLampedPinnedRod_sol_$(sol_number)_mode_$(rod_mode)_X$(xp_str)_Y$(yp_str).mat"
    
    safe_log_println("")
    safe_log_println("📊 Generated Files by Pipeline Steps:")
    safe_log_println("  🔸 Step 1 (Initial Rod Shape):")
    safe_log_println("    - Rod data: $(rod_filename)")
    safe_log_println("    - Figures: Rod_Configuration_plots_[timestamp]/ (folder)")
    
    safe_log_println("  🔸 Step 2 (Rotation Learning Data):")
    safe_log_println("    - Full dataset: $(full_dataset_filename) (saved in Learning DataSet folder)")
    
    safe_log_println("  🔸 Step 3 (Training/Testing Split):")
    safe_log_println("    • Training Set ($(round(Int, config.train_ratio*100))%):")
    safe_log_println("      - MATLAB format: $(base_filename)_train_$(train_ratio_str).mat")
    safe_log_println("      - Julia format:  $(base_filename)_train_$(train_ratio_str).jld2")
    safe_log_println("    • Testing Set ($(round(Int, (1-config.train_ratio)*100))%):")
    safe_log_println("      - MATLAB format: $(base_filename)_test_$(test_ratio_str).mat")
    safe_log_println("      - Julia format:  $(base_filename)_test_$(test_ratio_str).jld2")
    
    # Finalize logging
    if log_capture !== nothing
        finalize_logging(log_capture)
    end
    
    return true
end

# Run example only when file is executed directly (not when included)
# NOTE: This auto-execution is disabled when used in pipeline scripts
const THIS_FILE = @__FILE__
if @isdefined(__PIPELINE_ALREADY_LOADED__) && (abspath(PROGRAM_FILE) == THIS_FILE || FORCE_STANDALONE_EXECUTION)
    println("Starting complete rod solver pipeline with logging...")
    
    # Create custom config for xp = 0.2 (with logging enabled by default)
    custom_config = create_config(xp = 0.2, yp = 0.0, mode = 2, train_ratio = 0.85)

    # Run the complete pipeline with logging enabled
    # Note: Complete REPL capture is now handled globally at the top of the script
    # The pipeline logging provides structured logging in addition to the complete capture
    success = solve_and_prepare_data(custom_config, enable_logging=true, log_dir="logs", capture_all_output=true)

    # Simple final status
    if success
        println("\n" * "="^60)
        println("✓ PIPELINE COMPLETED: Rod solver finished for xp = $(custom_config.xp) and yp = $(custom_config.yp)")
        println("📝 Check the logs/ directory for detailed execution log")
        println("="^60)
    else
        println("\n" * "="^60)
        println("✗ PIPELINE FAILED - Check error messages above and log file")
        println("="^60)
    end

# If you want to see when the module is loaded silently, uncomment the next 3 lines:
# else
#     println(" Module loaded: solve_and_prepare_data functions available")
#     println(" (Script execution skipped - call solve_and_prepare_data() manually for custom configs)")
end

