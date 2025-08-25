# ---------------------------------------------------------------------------
# Dataset splitting utilities for training/testing separation
# ---------------------------------------------------------------------------

using MLUtils  # For splitting data
using JLD2
using MATLAB
using Dates
using Random  # For reproducible seeding

# Setup for standalone usage (when run directly)
if !@isdefined(ClampedRodConfig)
    include("../utils/config.jl")
end

"""
    split_dataset_for_training(config::ClampedRodConfig)

Split the generated DataSet_temp into training and testing sets based on the train_ratio.
Saves the split datasets as both .mat files (for MATLAB compatibility) and .jld2 files (for Julia).

# Arguments
- `config::ClampedRodConfig`: Rod configuration parameters

# Returns
- `Bool`: `true` if splitting and saving completed successfully, `false` otherwise

# Files Generated
For config with xp=X, yp=Y, mode=M, train_ratio=TR:
- Training set: `LearnigData_Rod_ClampedPinned_Rotated_X{X}_Y{Y}_mode{M}_train_{TR}.mat` and `.jld2`
- Testing set: `LearnigData_Rod_ClampedPinned_Rotated_X{X}_Y{Y}_mode{M}_test_{1-TR}.mat` and `.jld2`

Example: For xp=0.2, yp=0.0, mode=2, train_ratio=0.85:
- Training: `LearnigData_Rod_ClampedPinned_Rotated_X02_Y00_mode2_train_085.jld2`
- Testing: `LearnigData_Rod_ClampedPinned_Rotated_X02_Y00_mode2_test_015.jld2`

# Example
```julia
custom_config = create_config(xp = 0.9, yp = 0.0, mode = 2.0)
success = split_dataset_for_training(custom_config)
```

# References
Based on work from University of the Basque Country UPV/EHU (Oscar Altuzarra, 2021 & 2025)
"""
function split_dataset_for_training(config)
    println("\n" * "="^60)
    println("DATASET SPLITTING FOR TRAINING/TESTING")
    println("="^60)
    
    println("Train ratio: $(config.train_ratio) ($(round(Int, config.train_ratio*100))% training, $(round(Int, (1-config.train_ratio)*100))% testing)")
    
    try
        # Since we use isolated process execution, load dataset directly from saved file
        println("\n🔄 Loading dataset from saved file...")
        
        # Generate filename components
        xp_str = replace(string(config.xp), "." => "", "-" => "neg")
        yp_str = replace(string(config.yp), "." => "", "-" => "neg")  
        mode_str = replace(string(Int(config.mode)), "." => "")
        
        # Construct the expected filename
        saved_filename = "LearnigData_Rod_ClampedPinned_Rotated_X$(xp_str)_Y$(yp_str)_72sols_mode$(mode_str)_revised.mat"
        saved_path = joinpath("dataset", "MATLAB code", "Learning_Data_ClampedPinned_Rod_IK", saved_filename)
        
        if !isfile(saved_path)
            println("✗ Cannot find saved dataset file: $(saved_path)")
            throw(ErrorException("Dataset file not found: $(saved_path)"))
        end
        
        println("   Loading dataset from: $(saved_path)")
        
        # Load dataset in MATLAB  
        mat"""
        clear DataSet_temp DataSet;
        load($saved_path);
        if exist('DataSet', 'var')
            DataSet_temp = DataSet;
            fprintf('✓ Dataset loaded: %d trajectories x %d points\\n', size(DataSet_temp, 1), size(DataSet_temp, 2));
        else
            error('DataSet variable not found in .mat file');
        end
        """
        
        # Retrieve the loaded dataset
        @mget DataSet_temp
        println("✓ Successfully loaded DataSet from saved file")
        
        # Add MATLAB path for the target save directory (repository-portable)
        project_root = pwd()
        target_dir_absolute = joinpath(project_root, "dataset", "MATLAB code", "Learning_Data_ClampedPinned_Rod_IK", "02.-Learning DataSet")
        target_dir_matlab = replace(target_dir_absolute, "\\" => "/")
        
        @mput target_dir_matlab
        
        mat"""
        if ~exist(target_dir_matlab, 'dir')
            mkdir(target_dir_matlab);
            fprintf('✓ Created target directory: %s\\n', target_dir_matlab);
        end
        cd(target_dir_matlab);
        fprintf('✓ Changed MATLAB working directory to: %s\\n', target_dir_matlab);
        """
        
        # Get dataset dimensions
        num_trajectories, num_points = size(DataSet_temp)
        println("Dataset size: $(num_trajectories) trajectories × $(num_points) data points")
        
        # Set random seed for reproducible splits
        Random.seed!(42)
        
        # Use MLUtils.splitobs with transposed data (observations in last dimension)
        # Transpose so splitobs works on columns (trajectories), then transpose back
        train_data_T, test_data_T = splitobs(DataSet_temp', at=config.train_ratio, shuffle=true)
        
        # Transpose back to have observations as rows (trajectories)
        DataSet_train = Matrix{Float64}(train_data_T')  # Convert to simple Matrix type
        DataSet_test = Matrix{Float64}(test_data_T')    # Convert to simple Matrix type
        
        # Get split sizes
        num_train, num_test = size(DataSet_train, 1), size(DataSet_test, 1)
        
        println("Split sizes:")
        println("  - Training: $(num_train) trajectories ($(round(num_train/num_trajectories*100, digits=1))%)")
        println("  - Testing:  $(num_test) trajectories ($(round(num_test/num_trajectories*100, digits=1))%)")
        
        println("\n💾 Saving split datasets...")
        
        # Generate filenames based on configuration
        xp_str = replace(string(config.xp), "." => "", "-" => "neg")
        yp_str = replace(string(config.yp), "." => "", "-" => "neg")
        mode_str = replace(string(Int(config.mode)), "." => "")
        
        # Format train/test ratios for filenames (remove decimal point, round to avoid floating point issues)
        train_ratio_str = replace(string(round(config.train_ratio, digits=3)), "." => "")
        test_ratio_str = replace(string(round(1.0 - config.train_ratio, digits=3)), "." => "")
        
        base_filename = "LearnigData_Rod_ClampedPinned_Rotated_X$(xp_str)_Y$(yp_str)_mode$(mode_str)"
        
        # Define target directory - use organized 02.-Learning DataSet folder
        target_dir = joinpath("dataset", "MATLAB code", "Learning_Data_ClampedPinned_Rod_IK", "02.-Learning DataSet")
        
        # Ensure target directory exists
        if !isdir(target_dir)
            mkpath(target_dir)
            println("✓ Created target directory: $(target_dir)")
        end
        
        # Create full file paths including target directory and train/test ratios
        train_filename_mat = joinpath(target_dir, "$(base_filename)_train_$(train_ratio_str).mat")
        test_filename_mat = joinpath(target_dir, "$(base_filename)_test_$(test_ratio_str).mat")
        train_filename_jld2 = joinpath(target_dir, "$(base_filename)_train_$(train_ratio_str).jld2")
        test_filename_jld2 = joinpath(target_dir, "$(base_filename)_test_$(test_ratio_str).jld2")
        
        # Save training set
        println("  - Saving training set...")
        
        # Save as .mat file - MATLAB now has the target directory in its path
        train_filename_simple = "$(base_filename)_train_$(train_ratio_str).mat"
        @mput DataSet_train train_filename_simple
        
        mat"""
        save(train_filename_simple, 'DataSet_train');
        """
        println("    ✓ $(train_filename_mat)")
        
        # Save as .jld2 file (Julia format)
        jldsave(train_filename_jld2; 
                 DataSet_train = DataSet_train,
                 config = config,
                 num_trajectories = num_train,
                 split_info = "Training set - $(num_train) trajectories",
                 train_ratio = config.train_ratio,
                 timestamp = string(now()))
        println("    ✓ $(train_filename_jld2)")
        
        # Save testing set
        println("  - Saving testing set...")
        
        # Save as .mat file - MATLAB now has the target directory in its path
        test_filename_simple = "$(base_filename)_test_$(test_ratio_str).mat"
        @mput DataSet_test test_filename_simple
        
        mat"""
        save(test_filename_simple, 'DataSet_test');
        """
        println("    ✓ $(test_filename_mat)")
        
        # Save as .jld2 file (Julia format)
        jldsave(test_filename_jld2; 
                 DataSet_test = DataSet_test,
                 config = config,
                 num_trajectories = num_test,
                 split_info = "Testing set - $(num_test) trajectories",
                 train_ratio = config.train_ratio,
                 timestamp = string(now()))
        println("    ✓ $(test_filename_jld2)")
        
        println("\n" * "="^60)
        println("✓ DATASET SPLITTING COMPLETED SUCCESSFULLY!")
        println("="^60)
        println("Generated files:")
        println("  Training: $(train_filename_mat), $(train_filename_jld2)")
        println("  Testing:  $(test_filename_mat), $(test_filename_jld2)")
        
        return true
        
    catch e
        println("✗ Error during dataset splitting: $e")
        println("Make sure DataSet_temp is available in MATLAB workspace")
        return false
    end
end

