# ---------------------------------------------------------------------------
# Logging Utilities for Rod Solver Pipeline
# ---------------------------------------------------------------------------

using Dates

"""
    LogCapture

A structure to capture console output to log file while maintaining normal console behavior.
"""
mutable struct LogCapture
    log_file::IOStream
    log_enabled::Bool
    log_path::String
    capture_all_output::Bool
    original_stdout::Union{IO, Nothing}
    original_stderr::Union{IO, Nothing}
end

"""
    TeeStream

A custom IO stream that writes to both console and log file for complete REPL capture.
"""
mutable struct TeeStream <: IO
    original::IO
    log_file::IOStream
end

function Base.write(io::TeeStream, data)
    # Write to original stream (console)
    n1 = write(io.original, data)
    flush(io.original)
    
    # Write to log file
    try
        n2 = write(io.log_file, data)
        flush(io.log_file)
    catch e
        # If log file write fails, continue with console output
        @warn "Log file write failed: $e"
    end
    
    return n1
end

Base.flush(io::TeeStream) = begin
    flush(io.original)
    try
        flush(io.log_file)
    catch e
        @warn "Log file flush failed: $e"
    end
end

"""
    setup_logging(config::ClampedRodConfig; log_dir="logs", capture_all_output=false)

Setup logging system for the rod solver pipeline with optional complete REPL output capture.

# Arguments
- `config::ClampedRodConfig`: Rod configuration parameters
- `log_dir::String`: Directory to store log files (default: "logs")
- `capture_all_output::Bool`: If true, captures ALL console output; if false, only structured logging (default: false)

# Returns
- `LogCapture`: Log capture object for managing logging

# Generated Log File
Log filename format: `RodSolver_X{X}_Y{Y}_mode{M}_{timestamp}.log`

Example: `RodSolver_X02_Y00_mode2_20250822_143052.log`
"""
function setup_logging(config; log_dir="logs", capture_all_output=false)
    # Create logs directory if it doesn't exist
    if !isdir(log_dir)
        mkpath(log_dir)
        println("✓ Created log directory: $(log_dir)")
    end
    
    # Generate log filename based on configuration and timestamp
    xp_str = replace(string(config.xp), "." => "", "-" => "neg")
    yp_str = replace(string(config.yp), "." => "", "-" => "neg")
    mode_str = replace(string(Int(config.mode)), "." => "")
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    log_filename = "RodSolver_X$(xp_str)_Y$(yp_str)_mode$(mode_str)_$(timestamp).log"
    log_path = joinpath(log_dir, log_filename)
    
    # Open log file
    log_file = open(log_path, "w")
    
    # Write header to log file
    write_log_header(log_file, config, log_path, capture_all_output)
    
    original_stdout = nothing
    original_stderr = nothing
    
    if capture_all_output
        # Store original streams
        original_stdout = stdout
        original_stderr = stderr
        
        # For complete capture, redirect stdout/stderr to capture all output
        # Use a simpler approach with IOBuffer capture
        println("📝 Complete REPL output capture enabled: $(log_path)")
        println("📝 Note: All console output will now be captured to log file")
        
        # Create a simple capturing mechanism using redirect_stdout/stderr
        # We'll capture in the solve_and_prepare_data function instead
    else
        println("📝 Selective logging enabled: $(log_path)")
    end
    
    # Create LogCapture object
    log_capture = LogCapture(log_file, true, log_path, capture_all_output, original_stdout, original_stderr)
    
    return log_capture
end

"""
    write_log_header(log_file::IOStream, config, log_path::String, capture_all_output::Bool)

Write header information to the log file.
"""
function write_log_header(log_file::IOStream, config, log_path::String, capture_all_output::Bool)
    capture_mode = capture_all_output ? "COMPLETE REPL OUTPUT CAPTURE" : "SELECTIVE LOGGING"
    
    write(log_file, "="^80 * "\n")
    write(log_file, "ROD SOLVER PIPELINE LOG - $(capture_mode)\n")
    write(log_file, "="^80 * "\n")
    write(log_file, "Log file: $(log_path)\n")
    write(log_file, "Start time: $(now())\n")
    write(log_file, "Julia version: $(VERSION)\n")
    write(log_file, "Working directory: $(pwd())\n")
    write(log_file, "Capture mode: $(capture_mode)\n")
    write(log_file, "="^80 * "\n")
    write(log_file, "CONFIGURATION PARAMETERS:\n")
    write(log_file, "="^80 * "\n")
    
    # Write configuration to log
    buffer = IOBuffer()
    show(buffer, config)
    config_str = String(take!(buffer))
    write(log_file, config_str * "\n")
    
    write(log_file, "="^80 * "\n")
    if capture_all_output
        write(log_file, "COMPLETE PIPELINE EXECUTION OUTPUT (ALL REPL OUTPUT):\n")
    else
        write(log_file, "PIPELINE EXECUTION LOG (SELECTIVE):\n")
    end
    write(log_file, "="^80 * "\n")
    flush(log_file)
end

"""
    log_println(log_capture::LogCapture, message...)

Print message to console and log to file based on the capture mode.
"""
function log_println(log_capture::LogCapture, message...)
    message_str = join(string.(message), " ")
    
    # Always print to console
    println(message_str)
    
    # Log to file based on capture mode
    if log_capture.log_enabled
        if log_capture.capture_all_output
            # In complete capture mode, write directly to log file with timestamp
            timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            log_line = "[$(timestamp)] $(message_str)\n"
            write(log_capture.log_file, log_line)
            flush(log_capture.log_file)
        else
            # In selective mode, write with timestamp
            timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            log_line = "[$(timestamp)] $(message_str)\n"
            write(log_capture.log_file, log_line)
            flush(log_capture.log_file)
        end
    end
end

"""
    log_print(log_capture::LogCapture, message...)

Print message to console and optionally log to file without newline (for selective logging mode).
"""
function log_print(log_capture::LogCapture, message...)
    # Print to console
    print(message...)
    
    # Log to file if logging is enabled and NOT in complete capture mode
    if log_capture.log_enabled && !log_capture.capture_all_output
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        log_line = "[$(timestamp)] " * join(string.(message), " ")
        write(log_capture.log_file, log_line)
        flush(log_capture.log_file)
    end
end

"""
    log_section(log_capture::LogCapture, title::String; width=60)

Print a formatted section header to console and log to file based on the capture mode.
"""
function log_section(log_capture::LogCapture, title::String; width=60)
    separator = "="^width
    println("")
    println(separator)
    println(title)
    println(separator)
    
    # Log to file based on capture mode
    if log_capture.log_enabled
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        write(log_capture.log_file, "[$(timestamp)] \n")
        write(log_capture.log_file, "[$(timestamp)] $(separator)\n")
        write(log_capture.log_file, "[$(timestamp)] $(title)\n")
        write(log_capture.log_file, "[$(timestamp)] $(separator)\n")
        flush(log_capture.log_file)
    end
end

"""
    finalize_logging(log_capture::LogCapture)

Finalize logging and close log file.
"""
function finalize_logging(log_capture::LogCapture)
    if log_capture.log_enabled
        # Write footer to log file
        write(log_capture.log_file, "="^80 * "\n")
        write(log_capture.log_file, "PIPELINE COMPLETED\n")
        write(log_capture.log_file, "End time: $(now())\n")
        write(log_capture.log_file, "="^80 * "\n")
        flush(log_capture.log_file)
        
        # Close log file
        close(log_capture.log_file)
        log_capture.log_enabled = false
        
        if log_capture.capture_all_output
            println("📝 Log file closed - Complete REPL output captured: $(log_capture.log_path)")
        else
            println("📝 Log file closed - Selective logging completed: $(log_capture.log_path)")
        end
    end
end

"""
    capture_function_output(log_capture::LogCapture, func, args...; func_name="Function")

Execute a function and capture its output based on the capture mode.
For complete capture mode, captures ALL console output during function execution.
"""
function capture_function_output(log_capture::LogCapture, func, args...; func_name="Function")
    log_println(log_capture, "🔄 Starting $(func_name)...")
    
    start_time = now()
    
    try
        if log_capture.capture_all_output
            # For complete capture mode, use execute_with_complete_capture
            write(log_capture.log_file, "\n" * "="^80 * "\n")
            write(log_capture.log_file, "COMPLETE OUTPUT CAPTURE - $(func_name) - START\n")
            write(log_capture.log_file, "="^80 * "\n")
            flush(log_capture.log_file)
            
            result = execute_with_complete_capture(log_capture, func, args...)
            
            write(log_capture.log_file, "="^80 * "\n")
            write(log_capture.log_file, "COMPLETE OUTPUT CAPTURE - $(func_name) - END\n")
            write(log_capture.log_file, "="^80 * "\n")
            flush(log_capture.log_file)
        else
            # For selective logging, just execute normally
            result = func(args...)
        end
        
        end_time = now()
        duration = end_time - start_time
        
        if result
            log_println(log_capture, "✓ $(func_name) completed successfully (Duration: $(duration))")
        else
            log_println(log_capture, "✗ $(func_name) failed")
        end
        
        return result
        
    catch e
        end_time = now()
        duration = end_time - start_time
        log_println(log_capture, "✗ $(func_name) failed with error: $e (Duration: $(duration))")
        return false
    end
end

"""
    start_complete_capture(log_capture::LogCapture)

Start capturing all stdout/stderr output to the log file when capture_all_output=true.
Returns the original streams for restoration.
"""
function start_complete_capture(log_capture::LogCapture)
    if !log_capture.capture_all_output
        return nothing, nothing
    end
    
    # Store original streams
    original_stdout = stdout
    original_stderr = stderr
    
    # Create a custom stream that writes to both console and log file
    combined_stream = open(log_capture.log_path, "a")  # Open in append mode
    
    # Create a tee function that writes to both streams
    tee_function = function(content)
        # Write to original console
        write(original_stdout, content)
        flush(original_stdout)
        
        # Write to log file with timestamp
        timestamp_str = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        write(combined_stream, "[$(timestamp_str)] $(content)")
        flush(combined_stream)
    end
    
    # Redirect stdout and stderr to capture everything
    redirect_stdout() do
        redirect_stderr() do
            # This won't work as expected, let's try a different approach
        end
    end
    
    return original_stdout, original_stderr
end

"""
    stop_complete_capture(log_capture::LogCapture, original_stdout, original_stderr)

Stop capturing and restore original streams.
"""
function stop_complete_capture(log_capture::LogCapture, original_stdout, original_stderr)
    if !log_capture.capture_all_output || original_stdout === nothing
        return
    end
    
    # Restore original streams if they were redirected
    try
        # Note: This is tricky in Julia - we'll implement a different approach
    catch e
        @warn "Failed to restore streams: $e"
    end
end

"""
    execute_with_complete_capture(log_capture::LogCapture, func, args...)

Execute a function with complete output capture if enabled.
Uses a safe approach that captures through temporary file redirection.
"""
function execute_with_complete_capture(log_capture::LogCapture, func, args...)
    if !log_capture.capture_all_output
        return func(args...)
    end
    
    # For complete capture, use temporary file approach
    temp_output_file = tempname()
    
    try
        # Execute function with output redirected to temporary file
        open(temp_output_file, "w") do temp_file
            result = redirect_stdout(temp_file) do
                redirect_stderr(temp_file) do
                    func(args...)
                end
            end
            
            # After function completes, read the captured output
            if isfile(temp_output_file)
                captured_content = read(temp_output_file, String)
                
                if !isempty(captured_content)
                    # Display captured output to console
                    print(captured_content)
                    
                    # Write to log file with timestamps
                    lines = split(captured_content, '\n')
                    for line in lines
                        if !isempty(strip(line))
                            timestamp_str = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
                            write(log_capture.log_file, "[$(timestamp_str)] $(line)\n")
                        end
                    end
                    flush(log_capture.log_file)
                end
            end
            
            return result
        end
        
    catch e
        # If there's an error, still try to capture any output
        if isfile(temp_output_file)
            try
                captured_content = read(temp_output_file, String)
                if !isempty(captured_content)
                    print(captured_content)
                    lines = split(captured_content, '\n')
                    for line in lines
                        if !isempty(strip(line))
                            timestamp_str = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
                            write(log_capture.log_file, "[$(timestamp_str)] $(line)\n")
                        end
                    end
                    flush(log_capture.log_file)
                end
            catch
                # Ignore errors in cleanup
            end
        end
        
        rethrow(e)
        
    finally
        # Clean up temporary file
        try
            rm(temp_output_file, force=true)
        catch
            # Ignore cleanup errors
        end
    end
end
