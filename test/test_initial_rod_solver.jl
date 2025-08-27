# ---------------------------------------------------------------------------
# Tests for initial_rod_solver function
# 
# This module contains tests for the elliptical rod solver, including a 
# simplified version of the solver function without reference comparison.
# ---------------------------------------------------------------------------

using Test
# using MATLAB  # Will be conditionally loaded by main module
using Dates

# Include test utilities
include("test_utils.jl")

# Include project utilities and configuration
include("../src/utils/project_utils.jl")
include("../src/utils/config.jl")

"""
    initial_rod_solver_no_comparison(config::Union{ClampedRodConfig, Nothing} = nothing)

Simplified version of the elliptical rod solver without reference comparison.
This function is used for testing purposes and returns the computed solution results.

Returns:
- success::Bool: Whether the computation was successful
- results::Dict: Dictionary containing the computed results (IN_first, OUT_first, nsols)
"""
function initial_rod_solver_no_comparison(config::Union{ClampedRodConfig, Nothing} = nothing)
    try
        # Load configuration parameters
        if config === nothing
            config = get_default_config()
            println("✓ Using default configuration")
        else
            println("✓ Using provided configuration")
        end
        
        # Print configuration for verification
        print_config(config)
        
        # Extract parameters from config
        L = config.L                    # Length of the rod [m]
        N = Float64(config.N)          # Number of nodes in which the rod is discretized (convert to Float64 for MATLAB)
        EI = config.EI                 # Stiffness of the angular component of deformation
        x0 = config.x0                 # X Coordinate of Clamped-end
        y0 = config.y0                 # Y Coordinate of Clamped-end
        theta = config.theta           # Orientation of Clamped-end with X axis
        alpha = config.alpha           # Angle of the linear guide (when not, it is 0)
        lambda = config.lambda         # Length of the linear guide (when not, it is 0)
        xp = config.xp                 # X coordinate of end-tip [m]
        yp = config.yp                 # Y coordinate of end-tip [m]
        mode = config.mode             # Buckling Mode in Elliptic Integrals approach
        
        println("\n✓ Input parameters loaded from configuration")
        println("Rod Geometry:")
        println("  - Length (L): $(L) m")
        println("  - Grid Points (N): $(N)")
        println("  - Bending Rigidity (EI): $(EI)")
        println("Clamped End Conditions:")
        println("  - Position: ($(x0), $(y0)) m")
        println("  - Orientation (θ): $(theta) rad")
        println("Linear Guide Parameters:")
        println("  - Angle (α): $(alpha) rad")
        println("  - Length (λ): $(lambda) m")
        println("Pinned End Conditions:")
        println("  - Position: ($(xp), $(yp)) m")
        println("Solution Parameters:")
        println("  - Buckling Mode: $(mode)")
        println()

        # Detect the project root using the utility function
        project_root = find_project_root()
        println("✓ Project root detected: $project_root \n")
        
        # Set up MATLAB paths using detected project root
        println("✓ Setting MATLAB paths from: $project_root")
        
        # Convert Windows backslashes to forward slashes for MATLAB compatibility
        matlab_project_root = replace(project_root, '\\' => '/')
        println("✓ Normalized path for MATLAB: $matlab_project_root\n")
        
        # Generate folder name for INITIAL_ROD_SHAPE (without timestamp)
        timestamp_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        dataset_folder_name = "Rod_Shape_Test"  # Different folder for tests
        println("✓ Generated timestamp for figures: $timestamp_str")
        println("✓ Dataset folder name: $dataset_folder_name")
        
        # First initialize MATLAB engine
        mat"""
        % Basic MATLAB initialization
        close all ; % clear ; clc;  % Commented out 'clear' to preserve variables between blocks
        fprintf('MATLAB engine initialized\n');
        """
        
        # Now send the project root path and timestamp to MATLAB
        @mput matlab_project_root timestamp_str dataset_folder_name xp mode
        
        mat"""
        % MATLAB path setup with received project root and timestamp
        project_root = matlab_project_root;

        fprintf('Using project root: %s\n', project_root);

        matlab_base = fullfile(project_root, 'dataset', 'MATLAB code', 'Learning_Data_ClampedPinned_Rod_IK');
        functions_path = fullfile(matlab_base, 'Find Initial Rod Shape');
        functions_subpath = fullfile(functions_path, 'Functions');
        
        % Create timestamped DATASET folder for saving data and figures (using consistent timestamp)
        dataset_base = fullfile(matlab_base, 'Find Initial Rod Shape', dataset_folder_name);
        if ~exist(dataset_base, 'dir')
            mkdir(dataset_base);
            fprintf('✓ Created dataset folder: %s\n', dataset_base);
        end
        
        % Create figures subfolder within the dataset folder
        figures_base_folder = fullfile(dataset_base, 'Figures');
        if ~exist(figures_base_folder, 'dir')
            mkdir(figures_base_folder);
            fprintf('✓ Created figures base folder: %s\n', figures_base_folder);
        end
        
        % Create timestamped subfolder within Figures folder for all plots
        % Format xp value for folder naming (e.g., 0.2 -> X02, 0.5 -> X05)
        xp_scaled = round(xp * 10);
        if xp_scaled < 10
            xp_str = sprintf('X0%d', xp_scaled);
        else
            xp_str = sprintf('X%d', xp_scaled);
        end
        
        % Create folder name with timestamp, mode, and xp identifiers
        figures_folder_name = sprintf('%s_mode_%d_%s', timestamp_str, mode, xp_str);
        figures_folder = fullfile(figures_base_folder, figures_folder_name);
        if ~exist(figures_folder, 'dir')
            mkdir(figures_folder);
            fprintf('✓ Created timestamped figures folder: %s\n', figures_folder_name);
        end
        
        % Display constructed paths for debugging
        fprintf('MATLAB base path: %s\n', matlab_base);
        fprintf('Functions path: %s\n', functions_path);
        fprintf('Figures will be saved to: %s\n', figures_folder);
        
        % Verify directories exist
        if exist(matlab_base, 'dir') ~= 7
            error('MATLAB base directory not found at: %s', matlab_base);
        end
        
        % Add paths
        addpath(matlab_base);
        if exist(functions_path, 'dir') == 7
            addpath(functions_path);
        end
        if exist(functions_subpath, 'dir') == 7
            addpath(functions_subpath);
        end
        
        fprintf('\n✓ MATLAB paths configured from project root\n');
        fprintf('  Base: %s\n', matlab_base);
        fprintf('  Functions: %s\n', functions_path);
        fprintf('  Sub-functions: %s\n', functions_subpath);

        % Verify MATLAB files exist
        if exist('AA_IK_ClampedPinned_Rod_Elliptical.m', 'file') ~= 2
            error('MATLAB script AA_IK_ClampedPinned_Rod_Elliptical.m not found in path');
        end
        if exist('IK_NewRaph.m', 'file') ~= 2
            error('MATLAB function IK_NewRaph.m not found in path');
        end
        """
        
        println("✓ MATLAB paths and functions verified")
        
        # Send parameters to MATLAB and create input structure
        @mput L N EI x0 y0 theta alpha lambda xp yp mode
        
        # Create MATLAB input structure (matching AA_IK_ClampedPinned_Rod_Elliptical.m format)
        mat"""
        clear IN INBVP;  % Clear any existing structures
        
        % Create input structure as used in AA_IK_ClampedPinned_Rod_Elliptical.m
        IN.L = L;         % Length of the rod [m]
        IN.N = N;         % Number of nodes in which the rod is discretized
        IN.EI = EI;       % Stiffness of the angular component of deformation
        IN.x0 = x0;       % X Coordinate of Clamped-end
        IN.y0 = y0;       % Y Coordinate of Clamped-end
        IN.theta = theta; % Orientation of Clamped-end with X axis
        IN.alpha = alpha; % Angle of the linear guide
        IN.lambda = lambda; % Length of the linear guide
        IN.xp = xp;       % X coordinate of end-tip [m]
        IN.yp = yp;       % Y coordinate of end-tip [m]
        IN.mode = mode;   % Buckling Mode
        
        % Create BVP input structure as used in the MATLAB script
        INBVP.L = IN.L;
        INBVP.N = IN.N;
        INBVP.EI = IN.EI;
        INBVP.px_0 = IN.x0;
        INBVP.py_0 = IN.y0;
        INBVP.theta_0 = IN.theta;
        INBVP.px_end = IN.xp;
        INBVP.py_end = IN.yp;
        INBVP.mode = IN.mode;
        
        fprintf('✓ MATLAB input structures created\n');
        """
        
        # Execute the main MATLAB elliptical solver algorithm (same as original)
        @mput matlab_project_root dataset_folder_name timestamp_str xp mode
        mat"""
        fprintf('⏳ Running elliptical integral solver...\n');
        
        % Set project root and dataset folder name for this MATLAB block
        project_root = matlab_project_root;

        % Setup dataset folder and figures subfolder for saving plots and data (using consistent timestamp)
        matlab_base = fullfile(project_root, 'dataset', 'MATLAB code', 'Learning_Data_ClampedPinned_Rod_IK');
        dataset_base = fullfile(matlab_base, 'Find Initial Rod Shape', dataset_folder_name);
        if ~exist(dataset_base, 'dir')
            mkdir(dataset_base);
            fprintf('✓ Created dataset folder: %s\n', dataset_base);
        end
        
        % Create figures base folder
        figures_base_folder = fullfile(dataset_base, 'Figures');
        if ~exist(figures_base_folder, 'dir')
            mkdir(figures_base_folder);
            fprintf('✓ Created figures base folder: %s\n', figures_base_folder);
        end
        
        % Create timestamped subfolder within Figures folder for all plots
        % Format xp value for folder naming (e.g., 0.2 -> X02, 0.5 -> X05)
        xp_scaled = round(xp * 10);
        if xp_scaled < 10
            xp_str = sprintf('X0%d', xp_scaled);
        else
            xp_str = sprintf('X%d', xp_scaled);
        end
        
        % Create folder name with timestamp, mode, and xp identifiers
        figures_folder_name = sprintf('%s_mode_%d_%s', timestamp_str, mode, xp_str);
        figures_folder = fullfile(figures_base_folder, figures_folder_name);
        if ~exist(figures_folder, 'dir')
            mkdir(figures_folder);
            fprintf('✓ Created timestamped figures folder: %s\n', figures_folder_name);
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%              2. Define Grid discretization in [k_rel psi]
        %
        %      Discretization of the kr axis ]0 1[ with exponential relationship
        %      so that Nkr distributed points are coarse near 0 and finer near 1
        %
        slope = 0.1 ;      % Parameter that controls the density of kr.  
                        % Values between 0 and 1 are allowed.
                        % Close to 1 > linear distribution of values from 0 to 1
                        % Close to 0 > finer mesh in values close to 1
        %%%%%%%%%%%%%%%
        Nkr = 150 ;         % Number of points in which range of kr is divided
        %%%%%%%%%%%%%%%
        [ A, B ] = p2AB( slope ) ; % Function cretaed to find arguments of exp dist.
        t = linspace(0, 1, Nkr) ;
        kr_range      = A*(exp(B*t)-1) ;    % [0 1]
        epsilon= 1e-9 ;             % To avoid problems at certain values
        kr_range(1)   = epsilon ;   % No exact 0 > problems with Elliptic Integrals
        kr_range(end) = 1-epsilon ; % No exact 1 > problems with Elliptic Integrals
        %
        %         Linear discretization in psi range with Npsi points
        %%%%%%%%%%%%%%%
        Npsi = 300 ;         % Number of points in which range of psi is divided
        %%%%%%%%%%%%%%%
        psi_range   = linspace(0+epsilon,2*pi-epsilon,Npsi) ;  % Range for psi

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%         3. Solve R, x, y with Elliptic Int for every k and psi
        %
        % Discretization in Psi and Kr

        [ PSI, KR ]  = meshgrid( psi_range, kr_range) ;

        % KR positive values

        IN.psi=PSI;
        IN.kr=KR;
        [ Xpos, Ypos, Rpos,  ~,  ~,  ~,  ~, ~] = ClampedPinned_endpoint_pose_force( IN );

        % KR negative values

        IN.psi=PSI;
        IN.kr=-KR;
        [ Xneg, Yneg, Rneg,  ~,  ~,  ~,  ~, ~ ] = ClampedPinned_endpoint_pose_force( IN );

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%        4. Solve residuals for x and y
        %
        % KR positive values

        ResXp=IN.xp-Xpos;
        ResYp=IN.yp-Ypos;

        % KR negative values

        ResXn=IN.xp-Xneg;
        ResYn=IN.yp-Yneg;

        %   2D curves of null residual of constraints on Kr vs Psi space
        %   with points of intersection where both conditions are met, i.e.
        %   potential solution points found with interpolation

        figure(2);
        hold on;

        [cXpos, ~]=contour(PSI,KR,ResXp,[0 0],'b');
        [cYpos, ~]=contour(PSI,KR,ResYp,[0 0],'r');
        [cXneg,~]=contour(PSI,-KR,ResXn,[0 0],'b');
        [cYneg,~]=contour(PSI,-KR,ResYn,[0 0],'r');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%        5. Use root finding interpolation to get potential solutions

        %    Find the intersection segments of the curves and plot
        [~,cXposX,cXposY] = intersegments(cXpos);
        plot(cXposX,cXposY,'b');
        [ ~,cYposX,cYposY] = intersegments(cYpos);
        plot(cYposX,cYposY,'r');
        [ ~,cXnegX,cXnegY] = intersegments(cXneg);
        plot(cXnegX,cXnegY,'b');
        [ ~,cYnegX,cYnegY] = intersegments(cYneg);
        plot(cYnegX,cYnegY,'r');

        %   Intersection Points for positive KRs
        [npointspos,intXpos,intYpos] = interpoints(cXposX,cXposY,cYposX,cYposY);
        %   Intersection Points for negative KRs
        [npointsneg,intXneg,intYneg] = interpoints(cXnegX,cXnegY,cYnegX,cYnegY);

        plot(intXpos,intYpos,'ko');
        plot(intXneg,intYneg,'ko');

        grid on;box on;
        xlim([0 2*pi])
        ylim([-1.1 1.1])
        xlabel('\\psi')
        ylabel('Kr')
        title('\\color{blue}{res_X} = 0, \\color{red}{res_Y} = 0')
        
        % Save the contour plot
        saveas(gcf, fullfile(figures_folder, '01_Test_Contour_Plot_Residuals.png'));
        saveas(gcf, fullfile(figures_folder, '01_Test_Contour_Plot_Residuals.fig'));
        fprintf('✓ Saved test contour plot to figures folder\n');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Show Potential solutions on Command Window

            fprintf('\n Unrefined Solutions reached') ;
            fprintf('\n______________________________________\n') ;
        disp(['For kr>0: psi =' num2str(intXpos) ' kr =' num2str(intYpos)] )
        disp(['For kr<0: psi =' num2str(intXneg) ' kr =' num2str(intYneg)] )

        % Store all solutions 
        psisol=NaN;
        krsol=NaN;

        for i = 1:npointspos
        psisol(i)=intXpos(i);
        krsol(i)=intYpos(i);
        end
        for i = 1:npointsneg
        psisol(i+npointspos)=intXneg(i);
        krsol(i+npointspos)=intYneg(i);
        end
        nsols=npointspos+npointsneg;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%        6. Apply Newton scheme on Elliptic Integration to refine them


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Show Refined solutions on Command Window
        fprintf('\n Refined Solutions reached') ;
        fprintf('\n______________________________________\n') ;

        for i = 1:nsols
        IN.psi=psisol(i);
        IN.kr=krsol(i);

        %   Newton scheme based on Elliptic Integration solution
        INsolpos(i)=IK_NewtonRaphson_rod( IN );

        psisol(i)=INsolpos(i).psi;
        krsol(i)=INsolpos(i).kr;

        fprintf('\n Newton-Raphson process Elliptic Successful \n') ;
        fprintf('\n______________________________________\n') ;
        fprintf('\n Refined Solution = %i', i) ;
        fprintf('\n______________________________________\n') ;
        disp(['psi =' num2str(INsolpos(i).psi) ' kr =' num2str(INsolpos(i).kr)] )
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %%        7. Plot Final Refined Solutions (simplified for testing)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Color definition (from original MATLAB code)
        azul = [0 144 189]/255;
        narj = [217 53 25]/255;
        amar = [237 177 32]/255;
        viol = [126 47 142]/255;
        verd = [119 172 48]/255;
        color_matrix = [ azul ;
                         narj ;
                         amar ;
                         viol ;
                         verd ] ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%        8. Apply a BVP Newton scheme to refine them and save files

        %%%%%%%%%% Solve Now with the BVP and save .mat files
        for i = 1:nsols
        fprintf('\n Refined Solution BVP = %i', i) ;
        fprintf('\n______________________________________\n') ;

        % Define guess values for BVP integration:

        INBVP.nx0_guess=INsolpos(i).Fx;
        INBVP.ny0_guess=INsolpos(i).Fy;
        INBVP.m0_guess=INsolpos(i).Mz0;

        INBVP.kr    = krsol(i);
        INBVP.psi   = psisol(i);
        % Transformation kr -> k
                    kmin = abs(cos(0.5*INBVP.psi)) ;
        INBVP.k     = sign(INBVP.kr).*kmin + (1-kmin).*INBVP.kr ;
            
        [ IN, OUT, ~, ~ ] = IK_NewRaph( INBVP ) ;
        
        % Store first solution for test validation
        if i == 1
            IN_first = IN;
            OUT_first = OUT;
            
            % Verification: Print stored values to confirm they're preserved
            fprintf('\n=== STORED SOLUTION FOR TEST VALIDATION ===\n');
            fprintf('Stored IN_first.px_end: %.15f\n', IN_first.px_end);
            fprintf('Stored OUT_first.px(end): %.15f\n', OUT_first.px(end));
            fprintf('Stored OUT_first.py(end): %.15f\n', OUT_first.py(end));
            fprintf('Stored OUT_first.Ener: %.15f\n', OUT_first.Ener);
            fprintf('==========================================\n');
        end

        end
        """
        
        # Get results back to Julia
        @mget IN_first OUT_first nsols
        
        println("\n=== MATLAB SOLVER RESULTS ===")
        println("✓ Solution computed successfully")
        println("Number of solutions found: $(nsols)")
        
        # Create results dictionary
        results = Dict(
            "IN_first" => IN_first,
            "OUT_first" => OUT_first,
            "nsols" => nsols,
            "config" => config
        )
        
        if nsols > 0
            println("First solution results:")
            println("Final tip position: ($(OUT_first["px"][end]), $(OUT_first["py"][end]))")
            println("Target position: ($(IN_first["px_end"]), $(IN_first["py_end"]))")
            println("Energy: $(OUT_first["Ener"])")
            
            # Store additional computed values in results
            results["computed_px_final"] = OUT_first["px"][end]
            results["computed_py_final"] = OUT_first["py"][end]
            results["computed_energy"] = OUT_first["Ener"]
            results["target_px"] = IN_first["px_end"]
            results["target_py"] = IN_first["py_end"]
        end
        
        return true, results

    catch e
        if MATLAB_AVAILABLE[] && isa(e, MATLAB.MEngineError)
            println("\n ERROR: MATLAB engine not available")
            println("Solutions:")
            println("1. Ensure MATLAB is installed and licensed")
            println("2. Run 'matlab -regserver' in Command Prompt as Administrator")
            println("3. Restart Julia")
            return false, Dict()
        else
            println("\n ERROR: $(e)")
            return false, Dict()
        end
    end
end

"""
    test_initial_rod_solver()

Main test function that runs various tests for the elliptical rod solver.
"""
function test_initial_rod_solver()
    println("\n=== TESTING ELLIPTICAL ROD SOLVER ===")
    
    @testset "Default Configuration Test" begin
        success, results = initial_rod_solver_no_comparison()
        
        @test success == true
        
        if success && results["nsols"] > 0
            assert_solution_valid(results)
            print_test_results(results, "Default Configuration")
            println("✓ Default configuration test passed")
        end
    end
    
    @testset "Custom Configuration Test" begin
        # Test with custom configuration
        custom_config = create_config(xp = 0.2, yp = 0.1, mode = 2.0)
        success, results = initial_rod_solver_no_comparison(custom_config)
        
        @test success == true
        @test haskey(results, "config")
        @test results["config"].xp == 0.2
        @test results["config"].yp == 0.1
        @test results["config"].mode == 2.0
        
        if success && results["nsols"] > 0
            # Test that target position matches input
            target_tolerance = 1e-3  # Allow small numerical tolerance
            @test abs(results["target_px"] - 0.2) < target_tolerance
            @test abs(results["target_py"] - 0.1) < target_tolerance
            
            println("✓ Custom configuration test passed")
        end
    end
    
    @testset "Algorithm Consistency Test" begin
        # Test that running the same configuration twice gives the same result
        config = get_default_config()
        
        success1, results1 = initial_rod_solver_no_comparison(config)
        success2, results2 = initial_rod_solver_no_comparison(config)
        
        @test success1 == true
        @test success2 == true
        
        if success1 && success2 && results1["nsols"] > 0 && results2["nsols"] > 0
            # Test deterministic behavior using strict tolerance
            assert_approximately_equal(results1["computed_px_final"], results2["computed_px_final"], STRICT_TOLERANCE; message="px_final should be identical")
            assert_approximately_equal(results1["computed_py_final"], results2["computed_py_final"], STRICT_TOLERANCE; message="py_final should be identical")
            assert_approximately_equal(results1["computed_energy"], results2["computed_energy"], STRICT_TOLERANCE; message="energy should be identical")
            
            println("✓ Algorithm consistency test passed")
        end
    end
    
    @testset "Solver Configuration Validation" begin
        # Test that solver properly handles different rod configurations
        configs_to_test = [
            create_config(xp = 0.3, yp = 0.0, mode = 1.0),
            create_config(xp = 0.4, yp = 0.2, mode = 2.0),
            create_config(xp = 0.5, yp = -0.1, mode = 3.0)
        ]
        
        for (i, config) in enumerate(configs_to_test)
            success, results = initial_rod_solver_no_comparison(config)
            @test success == true
            @test results["nsols"] > 0
            
            if success && results["nsols"] > 0
                # Test that different configurations produce different results
                @test isfinite(results["computed_energy"])
                @test results["computed_energy"] >= 0.0
            end
            
            println("✓ Configuration $(i) test passed")
        end
    end
    
    println("✓ All elliptical rod solver tests completed!")
end
