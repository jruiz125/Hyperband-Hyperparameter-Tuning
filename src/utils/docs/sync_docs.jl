"""
Documentation Auto-Sync Utilities for ClampFixedRodSolver

Simple functions to keep docs_for_language_server.jl synchronized with
exported functions from the main module.

Usage:
```julia
include("sync_docs.jl")
sync_documentation()  # Update missing functions
sync_documentation(force=true)  # Regenerate all documentation
```
"""

using Dates: now

"""
    sync_documentation(; force::Bool=false, verbose::Bool=true)

Automatically synchronize the documentation file with exported functions.

# Arguments
- `force::Bool=false`: If true, regenerates all documentation. If false, only adds missing functions.
- `verbose::Bool=true`: If true, prints progress information.

# Examples
```julia
# Add only missing functions
sync_documentation()

# Regenerate all documentation
sync_documentation(force=true)

# Silent mode
sync_documentation(verbose=false)
```
"""
function sync_documentation(; force::Bool=false, verbose::Bool=true)
    # File paths
    main_file = joinpath(@__DIR__, "..", "..", "ClampFixedRodSolver.jl")
    docs_file = joinpath(@__DIR__, "..", "docs_for_language_server.jl")
    
    if !isfile(main_file)
        error("Main module file not found: $main_file")
    end
    
    # Get exported symbols
    exported_symbols = get_exported_symbols(main_file)
    documented_symbols = isfile(docs_file) ? get_documented_symbols(docs_file) : Symbol[]
    
    if verbose
        println("Found $(length(exported_symbols)) exported symbols")
    end
    
    # Determine what needs to be added
    if force
        missing_symbols = exported_symbols
        if verbose
            println("Force mode: regenerating documentation for all symbols")
        end
    else
        missing_symbols = setdiff(exported_symbols, documented_symbols)
        if verbose && isempty(missing_symbols)
            println("✅ All exported symbols are already documented!")
            return true
        elseif verbose
            println("Adding documentation for $(length(missing_symbols)) missing symbols")
        end
    end
    
    # Generate documentation
    if force || !isfile(docs_file)
        create_documentation_file(docs_file, missing_symbols, verbose)
    else
        append_documentation(docs_file, missing_symbols, verbose)
    end
    
    # Test the file
    try
        include(docs_file)
        if verbose
            println("✅ Documentation file updated and loads successfully!")
        end
        return true
    catch e
        if verbose
            println("❌ Error loading documentation file: $e")
        end
        return false
    end
end

"""
Extract exported symbols from the main module file.
"""
function get_exported_symbols(main_file::String)
    content = read(main_file, String)
    exported_symbols = Symbol[]
    
    # Find export statements
    for line in split(content, '\n')
        line = strip(line)
        if startswith(line, "export ")
            # Remove "export " and split by commas
            exports_str = line[8:end]  # Remove "export "
            
            # Handle multi-line exports and comments
            exports_str = split(exports_str, '#')[1]  # Remove comments
            exports_str = replace(exports_str, r"\s+" => " ")  # Normalize whitespace
            
            for symbol_str in split(exports_str, ',')
                symbol_str = strip(symbol_str)
                if !isempty(symbol_str)
                    push!(exported_symbols, Symbol(symbol_str))
                end
            end
        end
    end
    
    return unique(exported_symbols)
end

"""
Extract documented symbols from the docs file.
"""
function get_documented_symbols(docs_file::String)
    content = read(docs_file, String)
    documented_symbols = Symbol[]
    
    # Find function definitions
    for line in split(content, '\n')
        line = strip(line)
        
        # Function definitions
        func_match = match(r"^function\s+([a-zA-Z_][a-zA-Z0-9_!]*)", line)
        if func_match !== nothing
            push!(documented_symbols, Symbol(func_match.captures[1]))
        end
        
        # Struct definitions
        struct_match = match(r"^(?:mutable\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_!]*)", line)
        if struct_match !== nothing
            push!(documented_symbols, Symbol(struct_match.captures[1]))
        end
    end
    
    return unique(documented_symbols)
end

"""
Generate template documentation for a symbol.
"""
function generate_template_doc(symbol_name::Symbol)
    # Check if it's likely a struct/type
    symbol_str = string(symbol_name)
    is_type = occursin(r"Config$|Type$", symbol_str) || occursin(r"^[A-Z]", symbol_str)
    
    if is_type
        return """
\"\"\"
    $symbol_name

[AUTO-GENERATED] Type/struct definition for the ClampFixedRodSolver package.

Please add detailed documentation here including:
- Purpose and usage
- Field descriptions (if applicable)
- Constructor information
- Examples

# Examples
```julia
# Add usage examples here
instance = $symbol_name()
```

# See Also
- Add related functions/types here
\"\"\"
$(occursin("Config", symbol_str) ? "mutable " : "")struct $symbol_name end
"""
    else
        return """
\"\"\"
    $symbol_name

[AUTO-GENERATED] Function from the ClampFixedRodSolver package.

Please add detailed documentation here including:
- Function purpose
- Parameter descriptions  
- Return value information
- Usage examples

# Arguments
- Add function arguments here

# Returns
- Add return value description here

# Examples
```julia
# Add usage examples here
result = $symbol_name()
```

# See Also
- Add related functions here
\"\"\"
function $symbol_name end
"""
    end
end

"""
Create a new documentation file.
"""
function create_documentation_file(docs_file::String, symbols::Vector{Symbol}, verbose::Bool)
    header = """
\"\"\"
Documentation stubs for ClampFixedRodSolver Language Server support.

This file provides function and type definitions that the Julia Language Server
can understand for hover documentation and IntelliSense features.

Auto-generated on $(now()) by sync_docs.jl
Last updated: $(now())

To update this file:
```julia
include("sync_docs.jl")
sync_documentation()
```
\"\"\"

"""
    
    entries = [generate_template_doc(sym) for sym in symbols]
    content = header * join(entries, "\n")
    
    write(docs_file, content)
    
    if verbose
        println("✅ Created documentation file with $(length(symbols)) entries")
        for sym in symbols
            println("  ✓ $sym")
        end
    end
end

"""
Append documentation to existing file.
"""
function append_documentation(docs_file::String, symbols::Vector{Symbol}, verbose::Bool)
    if isempty(symbols)
        return
    end
    
    existing_content = read(docs_file, String)
    
    # Add update timestamp comment
    timestamp_comment = "\n# Auto-updated on $(now())\n"
    
    new_entries = [generate_template_doc(sym) for sym in symbols]
    new_content = existing_content * timestamp_comment * join(new_entries, "\n")
    
    write(docs_file, new_content)
    
    if verbose
        println("✅ Added documentation for $(length(symbols)) new symbols:")
        for sym in symbols
            println("  ✓ $sym")
        end
    end
end

"""
    auto_sync_on_include()

Automatically sync documentation when this file is included.
Call this at the end of your main module or startup script.
"""
function auto_sync_on_include()
    sync_documentation(verbose=false)
end

# Export the main function
export sync_documentation, auto_sync_on_include

# Optionally auto-sync when this file is included (uncomment if desired)
# auto_sync_on_include()
