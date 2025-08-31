#!/usr/bin/env julia
"""
Quick Documentation Sync Script

Simple script to update docs_for_language_server.jl when you add new exported functions.

Usage:
    julia update_docs_simple.jl              # Add missing functions
    julia update_docs_simple.jl --force      # Regenerate all documentation
    julia update_docs_simple.jl --test       # Test current documentation
"""

# Include the sync functions
include("sync_docs.jl")

function main()
    args = ARGS
    
    if "--help" in args || "-h" in args
        println(__doc__)
        return
    end
    
    if "--test" in args
        # Test current documentation
        docs_file = "docs_for_language_server.jl"
        if isfile(docs_file)
            try
                include(docs_file)
                println("✅ Documentation file loads successfully!")
                
                # Show counts
                exported = get_exported_symbols("src/ClampedPinnedRodUDE.jl")
                documented = get_documented_symbols(docs_file)
                missing = setdiff(exported, documented)
                
                println("📊 Status:")
                println("  Exported functions: $(length(exported))")
                println("  Documented functions: $(length(documented))")
                println("  Missing documentation: $(length(missing))")
                
                if !isempty(missing)
                    println("  Missing: $(join(missing, ", "))")
                end
            catch e
                println("❌ Documentation file has errors: $e")
            end
        else
            println("❌ Documentation file not found: $docs_file")
        end
        return
    end
    
    force = "--force" in args
    
    println("🔄 Updating ClampedPinnedRodUDE documentation...")
    
    success = sync_documentation(force=force, verbose=true)
    
    if success
        println("🎉 Documentation update completed successfully!")
    else
        println("❌ Documentation update failed!")
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
