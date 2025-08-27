# Documentation Auto-Sync System

This directory (`src/utils/docs/`) contains an automated system to keep `src/utils/docs_for_language_server.jl` synchronized with exported functions from your ClampedPinnedRodSolver package.

## 📁 Files

- **`../docs_for_language_server.jl`** - Main documentation file for Language Server hover support
- **`sync_docs.jl`** - Core synchronization functions  
- **`update_docs_simple.jl`** - Simple CLI script for manual updates
- **`auto_update_docs.jl`** - One-liner for automatic updates
- **`update_docs.jl`** - Advanced update script with source extractiontion Auto-Sync System

This directory (`src/utils/docs/`) contains an automated system to keep `docs_for_language_server.jl` synchronized with exported functions from your ClampedPinnedRodSolver package.

## 📁 Files

- **`../docs_for_language_server.jl`** - Main documentation file for Language Server hover support
- **`sync_docs.jl`** - Core synchronization functions  
- **`update_docs_simple.jl`** - Simple CLI script for manual updates
- **`auto_update_docs.jl`** - One-liner for automatic updates
- **`update_docs.jl`** - Advanced update script with source extraction

## 🚀 Usage Methods

### Method 1: From Julia REPL/Script
```julia
# Load and run sync (from project root)
include("src/utils/docs/sync_docs.jl")
sync_documentation()  # Add missing functions only
sync_documentation(force=true)  # Regenerate all documentation
```

### Method 2: Using the Package Function
```julia
using ClampedPinnedRodSolver
ClampedPinnedRodSolver.sync_docs()  # Add missing functions
ClampedPinnedRodSolver.sync_docs(force=true)  # Regenerate all
```

### Method 3: Command Line
```bash
# From project root, run:
julia src/utils/docs/update_docs_simple.jl

# Force regeneration
julia src/utils/docs/update_docs_simple.jl --force

# Test current status  
julia src/utils/docs/update_docs_simple.jl --test
```

### Method 4: Auto-Update (One-liner)
Add this to any script for automatic background updates:
```julia
include("src/utils/docs/auto_update_docs.jl")
```

### Method 5: Manual Integration
Add to your startup script or module initialization:
```julia
# At the end of src/ClampedPinnedRodSolver.jl or in your startup
if isfile("src/utils/docs/sync_docs.jl")
    include("src/utils/docs/sync_docs.jl")
    sync_documentation(verbose=false)
end
```

## 🔄 When to Use

### Automatic Updates (Recommended)
- Add `include("auto_update_docs.jl")` to scripts that modify exported functions
- Add sync call to your testing workflow
- Include in package development startup

### Manual Updates
- When you add new exported functions
- When you want to regenerate all documentation
- When documentation gets out of sync

## 📝 What Gets Generated

When you add a new exported function, the system automatically creates:

### For Functions:
```julia
"""
    your_function_name

[AUTO-GENERATED] Function from the ClampedPinnedRodSolver package.

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
result = your_function_name()
```

# See Also
- Add related functions here
"""
function your_function_name end
```

### For Structs/Types:
```julia
"""
    YourStructName

[AUTO-GENERATED] Type/struct definition for the ClampedPinnedRodSolver package.

Please add detailed documentation here including:
- Purpose and usage
- Field descriptions (if applicable)
- Constructor information
- Examples

# Examples
```julia
# Add usage examples here
instance = YourStructName()
```

# See Also
- Add related functions/types here
"""
mutable struct YourStructName end
```

## ✏️ Customizing Generated Documentation

After running the sync, you should:

1. **Replace `[AUTO-GENERATED]` placeholders** with actual descriptions
2. **Add real parameter documentation** in the `# Arguments` section
3. **Provide actual usage examples** in the `# Examples` section
4. **Add cross-references** in the `# See Also` section
5. **Include implementation notes** as needed

## 🔍 Checking Status

```julia
# From command line
julia update_docs_simple.jl --test

# From Julia
include("sync_docs.jl")
exported = get_exported_symbols("src/ClampedPinnedRodSolver.jl")
documented = get_documented_symbols("docs_for_language_server.jl")
missing = setdiff(exported, documented)
println("Missing: ", missing)
```

## 🛠️ Advanced Usage

### Customizing Templates
Edit the `generate_template_doc()` function in `sync_docs.jl` to change the default documentation template.

### Adding Custom Documentation
You can manually add documentation for internal functions (not exported) by editing `docs_for_language_server.jl` directly.

### Integration with CI/CD
Add to your test workflow:
```julia
include("sync_docs.jl")
@assert sync_documentation(verbose=false) "Documentation sync failed"
```

## 🚨 Important Notes

1. **Don't edit auto-generated content directly** - it will be overwritten. Instead, customize the templates in `sync_docs.jl`.

2. **The documentation file is for Language Server only** - it doesn't affect your actual package documentation.

3. **Keep both files in sync** - if you rename/remove exported functions, manually clean up the documentation file.

4. **Test after updates** - the sync functions include validation to ensure the documentation file loads correctly.

## 🔧 Troubleshooting

### "Documentation file has errors"
- Check Julia syntax in `docs_for_language_server.jl`
- Look for unclosed strings or mismatched quotes
- Run `julia -e 'include("docs_for_language_server.jl")'` for specific error

### "No exported functions found"  
- Verify `src/ClampedPinnedRodSolver.jl` exists and has `export` statements
- Check file paths in sync functions

### "Auto-generated content not helpful"
- Customize the templates in `generate_template_doc()` function in `src/utils/docs/sync_docs.jl`
- Add better type detection logic for structs vs functions

## 📚 Example Workflow

1. Add new function to your source code:
```julia
# In src/solvers/new_solver.jl
function amazing_new_solver(x, y)
    # Implementation here
end
```

2. Export it in main module:
```julia
# In src/ClampedPinnedRodSolver.jl
export amazing_new_solver
```

3. Auto-update documentation:
```julia
julia utils/docs/update_docs_simple.jl
```

4. Customize the generated documentation:
```julia
# Edit docs_for_language_server.jl to replace [AUTO-GENERATED] content
```

5. Test hover functionality in VS Code! 🎉
