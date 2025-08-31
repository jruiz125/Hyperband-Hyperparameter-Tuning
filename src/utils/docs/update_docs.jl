#!/usr/bin/env julia
"""
Automatic Documentation Synchronizer for ClampedPinnedRodUDE

This script automatically updates docs_for_language_server.jl by:
1. Reading exported functions from src/ClampedPinnedRodUDE.jl
2. Comparing with existing documentation
3. Adding missing function stubs with template docstrings
4. Optionally extracting actual docstrings from source files

Usage:
    julia update_docs.jl                    # Update missing functions only
    julia update_docs.jl --extract-docs     # Also extract existing docstrings
    julia update_docs.jl --force            # Regenerate all documentation
"""

using Pkg
using Base: @__DIR__

# Configuration
const MAIN_MODULE_FILE = joinpath(@__DIR__, "..", "..", "ClampedPinnedRodUDE.jl")
const DOCS_FILE = joinpath(@__DIR__, "..", "docs_for_language_server.jl")
const SOURCE_DIRS = [
    joinpath(@__DIR__, "..", ".."),
    joinpath(@__DIR__, "..", "..", "solvers"),
    joinpath(@__DIR__, "..", "..", "utils"),
    joinpath(@__DIR__, "..", "..", "scripts")
]

"""
Extract exported symbols from the main module file.
"""
function get_exported_symbols()
    if !isfile(MAIN_MODULE_FILE)
        error("Main module file not found: $MAIN_MODULE_FILE")
    end
    
    content = read(MAIN_MODULE_FILE, String)
    exported_symbols = Symbol[]
    
    # Find all export statements
    export_matches = eachmatch(r"export\s+([^#\n]+)", content)
    
    for match in export_matches
        export_line = match.captures[1]
        # Split by comma and clean up whitespace
        symbols = split(export_line, ",")
        for symbol_str in symbols
            symbol_str = strip(symbol_str)
            if !isempty(symbol_str)
                push!(exported_symbols, Symbol(symbol_str))
            end
        end
    end
    
    return unique(exported_symbols)
end

"""
Extract documented symbols from the docs file.
"""
function get_documented_symbols()
    if !isfile(DOCS_FILE)
        return Symbol[]
    end
    
    content = read(DOCS_FILE, String)
    documented_symbols = Symbol[]
    
    # Find function definitions
    function_matches = eachmatch(r"^function\s+([a-zA-Z_][a-zA-Z0-9_!]*)", content, Base.RegexMatch)
    for match in function_matches
        push!(documented_symbols, Symbol(match.captures[1]))
    end
    
    # Find struct definitions
    struct_matches = eachmatch(r"^(?:mutable\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_!]*)", content, Base.RegexMatch)
    for match in struct_matches
        push!(documented_symbols, Symbol(match.captures[1]))
    end
    
    return unique(documented_symbols)
end

"""
Find the actual function definition in source files.
"""
function find_function_definition(func_name::Symbol)
    for dir in SOURCE_DIRS
        if !isdir(dir)
            continue
        end
        
        for file in readdir(dir, join=true)
            if endswith(file, ".jl") && isfile(file)
                content = read(file, String)
                
                # Look for function definition with various patterns
                patterns = [
                    r"^function\s+" * string(func_name) * r"\s*\([^)]*\)",
                    r"^" * string(func_name) * r"\s*\([^)]*\)\s*=",
                    r"^const\s+" * string(func_name) * r"\s*="
                ]
                
                for pattern in patterns
                    matches = eachmatch(pattern, content, Base.RegexMatch)
                    if !isempty(matches)
                        # Try to extract the full function signature
                        match = first(matches)
                        start_pos = match.offset
                        
                        # Find the complete signature (might span multiple lines)
                        lines = split(content, '\n')
                        line_starts = cumsum([0; length.(lines) .+ 1])
                        
                        line_num = findfirst(i -> line_starts[i] > start_pos, 1:length(line_starts)) - 1
                        if line_num === nothing
                            line_num = length(lines)
                        end
                        
                        # Extract signature (simplified)
                        signature_line = lines[line_num]
                        return (file, signature_line, line_num)
                    end
                end
            end
        end
    end
    
    return nothing
end

"""
Extract docstring for a function from source files.
"""
function extract_docstring(func_name::Symbol)
    definition_info = find_function_definition(func_name)
    if definition_info === nothing
        return nothing
    end
    
    file_path, signature_line, line_num = definition_info
    content = read(file_path, String)
    lines = split(content, '\n')
    
    # Look backwards from function definition to find docstring
    docstring_lines = String[]
    current_line = line_num - 1
    in_docstring = false
    
    while current_line >= 1
        line = strip(lines[current_line])
        
        if endswith(line, "\"\"\"") && !in_docstring
            # End of docstring found
            in_docstring = true
            if line != "\"\"\""
                # Docstring content on same line
                pushfirst!(docstring_lines, line[1:end-3])
            end
        elseif startswith(line, "\"\"\"") && in_docstring
            # Beginning of docstring found
            if line != "\"\"\""
                # Docstring content on same line
                pushfirst!(docstring_lines, line[4:end])
            end
            break
        elseif in_docstring
            pushfirst!(docstring_lines, line)
        elseif !isempty(line) && !startswith(line, "#")
            # Non-comment, non-empty line before function - no docstring
            break
        end
        
        current_line -= 1
    end
    
    if !isempty(docstring_lines)
        return join(docstring_lines, "\n")
    end
    
    return nothing
end

"""
Generate a template docstring for a function.
"""
function generate_template_docstring(func_name::Symbol)
    definition_info = find_function_definition(func_name)
    
    signature = "$(func_name)()"
    if definition_info !== nothing
        _, signature_line, _ = definition_info
        # Extract just the function signature part
        if occursin("function", signature_line)
            sig_match = match(r"function\s+(.+)", signature_line)
            if sig_match !== nothing
                signature = strip(sig_match.captures[1])
            end
        elseif occursin("=", signature_line)
            sig_match = match(r"(.+?)\s*=", signature_line)
            if sig_match !== nothing
                signature = strip(sig_match.captures[1])
            end
        end
    end
    
    return """
    $signature

[AUTO-GENERATED] Documentation for $(func_name).

# Description
This function is part of the ClampedPinnedRodUDE package.
Please add detailed documentation here.

# Arguments
- Add function arguments here

# Returns
- Add return value description here

# Examples
```julia
# Add usage examples here
result = $(func_name)()
```

# Notes
- Add implementation notes here
- This docstring was auto-generated and should be manually updated

# See Also
- Add related functions here
"""
end

"""
Generate documentation entry for a function or struct.
"""
function generate_doc_entry(symbol_name::Symbol, extract_existing::Bool = false)
    # Check if it's a struct by looking for definition
    definition_info = find_function_definition(symbol_name)
    is_struct = false
    
    if definition_info !== nothing
        _, signature_line, _ = definition_info
        is_struct = occursin(r"struct\s+", signature_line)
    end
    
    if is_struct
        return """
\"\"\"
    $(symbol_name)

[AUTO-GENERATED] Configuration struct.

Please add detailed documentation for this struct here.

# Fields
- Add struct fields here

# Examples
```julia
# Add usage examples here
instance = $(symbol_name)()
```
\"\"\"
$(occursin("Config", string(symbol_name)) ? "mutable " : "")struct $(symbol_name) end
"""
    else
        docstring = if extract_existing
            extracted = extract_docstring(symbol_name)
            extracted !== nothing ? extracted : generate_template_docstring(symbol_name)
        else
            generate_template_docstring(symbol_name)
        end
        
        return """
\"\"\"$(docstring)\"\"\"
function $(symbol_name) end
"""
    end
end

"""
Update the documentation file with missing functions.
"""
function update_documentation_file(extract_docs::Bool = false, force_regenerate::Bool = false)
    exported_symbols = get_exported_symbols()
    documented_symbols = get_documented_symbols()
    
    println("Found $(length(exported_symbols)) exported symbols:")
    for sym in exported_symbols
        println("  - $sym")
    end
    
    if force_regenerate
        println("\nForce regeneration mode - will recreate entire documentation file.")
        missing_symbols = exported_symbols
        documented_symbols = Symbol[]
    else
        missing_symbols = setdiff(exported_symbols, documented_symbols)
    end
    
    if isempty(missing_symbols)
        println("\n✅ All exported symbols are already documented!")
        return
    end
    
    println("\nMissing documentation for $(length(missing_symbols)) symbols:")
    for sym in missing_symbols
        println("  - $sym")
    end
    
    # Generate new documentation entries
    new_entries = String[]
    for symbol_name in missing_symbols
        println("Generating documentation for $symbol_name...")
        entry = generate_doc_entry(symbol_name, extract_docs)
        push!(new_entries, entry)
    end
    
    # Update the documentation file
    if force_regenerate || !isfile(DOCS_FILE)
        # Create new file
        header = """
\"\"\"
Documentation stubs for ClampedPinnedRodUDE Language Server support.

This file provides function and type definitions that the Julia Language Server
can understand for hover documentation and IntelliSense features.

Auto-generated on $(now()) by update_docs.jl
\"\"\"

"""
        
        content = header * join(new_entries, "\n")
        write(DOCS_FILE, content)
        println("\n✅ Created new documentation file: $DOCS_FILE")
    else
        # Append to existing file
        existing_content = read(DOCS_FILE, String)
        new_content = existing_content * "\n" * join(new_entries, "\n")
        write(DOCS_FILE, new_content)
        println("\n✅ Updated documentation file: $DOCS_FILE")
    end
    
    println("Added documentation for:")
    for sym in missing_symbols
        println("  ✓ $sym")
    end
end

"""
Main function - parse command line arguments and run update.
"""
function main()
    args = ARGS
    extract_docs = "--extract-docs" in args || "-e" in args
    force_regenerate = "--force" in args || "-f" in args
    
    if "--help" in args || "-h" in args
        println(__doc__)
        return
    end
    
    println("ClampedPinnedRodUDE Documentation Updater")
    println("=" ^ 50)
    
    try
        update_documentation_file(extract_docs, force_regenerate)
        
        # Test the updated file
        println("\nTesting updated documentation file...")
        include(DOCS_FILE)
        println("✅ Documentation file loads successfully!")
        
    catch e
        println("❌ Error: $e")
        rethrow(e)
    end
end

# Run if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
