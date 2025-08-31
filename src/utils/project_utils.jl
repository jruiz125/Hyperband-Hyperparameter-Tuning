"""
Project utilities for environment setup and path management
"""

using Pkg

"""
    find_project_root(; start_dir = @__DIR__)

Auto-detect project root directory by looking for characteristic files.

# Arguments
- `start_dir::String = @__DIR__`: Starting directory for search

# Returns
- `String`: Path to the project root directory

# Throws
- `ErrorException`: If project root cannot be found
"""
function find_project_root(; start_dir = @__DIR__) 
    current = start_dir
    while current != dirname(current)
        # Look for characteristic project files
        if isdir(joinpath(current, "src")) && 
           isfile(joinpath(current, "Project.toml"))
            return current
        end
        current = dirname(current)
    end
    error("Could not find project root. Make sure you're running from within the project structure.")
end

"""
    setup_project_environment(; activate_env = true, instantiate = false)

Sets up the project environment.

# Keyword Arguments
- `activate_env::Bool = true`: Whether to activate the Julia environment
- `instantiate::Bool = false`: Whether to run Pkg.instantiate() to install dependencies

# Returns
- `String`: Path to the project root
"""
function setup_project_environment(; activate_env = true, instantiate = false)
    project_root = find_project_root()
    cd(project_root)
    println("Project root detected at: ", pwd(),"\n")
    
    if activate_env
        Pkg.activate(project_root)
        println("\n✓ Environment activated\n")
        
        if instantiate
            println("⏳ Installing dependencies...")
            Pkg.instantiate()
            println("✓ Dependencies installed")
        end
    end
    
    return project_root
end