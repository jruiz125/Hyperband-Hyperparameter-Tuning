# Quick Documentation Auto-Update
# Add this line to any script to automatically update documentation
include(joinpath(@__DIR__, "sync_docs.jl")); sync_documentation(verbose=false)
