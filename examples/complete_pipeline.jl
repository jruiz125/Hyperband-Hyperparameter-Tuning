using Pkg
Pkg.add(url="https://github.com/jruiz125/Clamped-Pinned-Rod-Solver.git")
using ClampFixedRodSolver

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

# Run complete 3-step pipeline:
    # 1. Generate initial rod shape (initial_rod_solver)
    # 2. Generate rotation learning data (clamp_fixed_rod_solver) 
    # 3. Split dataset for training/testing (dataset_splitter)
success = solve_and_prepare_data(config)

if success
    println("✅ Pipeline completed successfully!")
    println("📊 Training/testing datasets ready for machine learning")
    println("📁 Check dataset/MATLAB code/Learning_Data_ClampedPinned_Rod_IK/02.-Learning DataSet/")
else
    println("❌ Pipeline failed - check logs for details")
end



