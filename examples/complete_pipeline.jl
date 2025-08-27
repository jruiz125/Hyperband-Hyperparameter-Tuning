using Pkg
Pkg.add(url="https://github.com/jruiz125/Clamped-Pinned-Rod-Solver.git")
using ClampedPinnedRodSolver

# Setup project environment
project_root = setup_project_environment(activate_env=true)

# Create configuration
config = create_config(
    xp = 0.2,           # Target x position
    yp = 0.0,           # Target y position
    mode = 2,           # Buckling mode
    train_ratio = 0.85, # 85% training, 15% testing
    save_figures = true # Enable figure saving
)

# ===================================================================
# 📖 FUNCTION DOCUMENTATION & HOVER SUPPORT
# ===================================================================
# Due to MATLAB dependency conflicts, function hover/IntelliSense may not work
# properly in this file. For full documentation with hover support:
#
# 1. Open: src/utils/docs_for_language_server.jl
# 2. Hover over function names for complete documentation
# 3. This file contains all function signatures without MATLAB dependencies
#
# Auto-open documentation file (uncomment if VS Code is your editor):
run(`code "src/utils/docs_for_language_server.jl"`)

println("📖 For function documentation with hover support:")
println("   Open: src/utils/docs_for_language_server.jl")
println("   This file provides full IntelliSense without MATLAB conflicts")

# Run complete 3-step pipeline:
    # 1. Generate initial rod shape (initial_rod_solver)
    # 2. Generate rotation learning data (clamp_fixed_rod_solver) 
    # 3. Split dataset for training/testing (dataset_splitter)
success = solve_and_prepare_data(config, capture_all_output = true)

if success
    println("✅ Pipeline completed successfully!")
    println("📊 Training/testing datasets ready for machine learning")
    println("📁 Check dataset/MATLAB code/Learning_Data_ClampedPinned_Rod_IK/Learning DataSet/")
else
    println("❌ Pipeline failed - check logs for details")
end



