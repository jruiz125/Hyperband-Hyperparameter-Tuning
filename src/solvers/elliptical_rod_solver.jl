# ---------------------------------------------------------------------------
# Elliptical Rod Solver
#
# Solves the inverse position problem using elliptic integrals
# 
# Can be run standalone or as part of the EllipticalRodSolver module
# ---------------------------------------------------------------------------

# Setup for standalone usage (when run directly)
if !@isdefined(ClampedRodConfig)
    include("../utils/project_utils.jl")
    include("../utils/config.jl")
end

# Import required packages  
using Dates

# Julia function to call MATLAB elliptical rod solver, AA_IK_ClampedPinned_Rod_Elliptical.m, and compare results
"""
    elliptical_rod_solver(config::Union{ClampedRodConfig, Nothing} = nothing)

Solves the inverse position problem of a clamped-pinned elastic rod using elliptic integrals.

This function interfaces with MATLAB to solve the inverse kinematics problem for a flexible rod
that is clamped at one end (fixed position and orientation) and pinned at the other end 
(fixed position, free orientation). Given desired tip coordinates, it finds all possible 
rod configurations using an elliptic integral approach.

# Arguments
- `config::Union{ClampedRodConfig, Nothing} = nothing`: Rod configuration parameters. 
  If `nothing`, uses default configuration from `get_default_config()`.

# Algorithm Steps
1. **Problem Setup**: Load configuration and initialize MATLAB environment
2. **Grid Discretization**: Create parameter space grid in (ψ, kr) coordinates  
3. **Elliptic Integration**: Solve rod endpoint positions for each grid point
4. **Residual Analysis**: Compute position errors and find zero-residual contours
5. **Root Finding**: Locate intersection points as potential solutions
6. **Newton Refinement**: Improve solution accuracy using Newton-Raphson method
7. **Visualization**: Generate plots of rod configurations and save figures
8. **BVP Verification**: Validate solutions using Boundary Value Problem solver

# Returns
- `Bool`: `true` if solver completed successfully, `false` if errors occurred

# Output Files
Generated in timestamped folders under `dataset/MATLAB code/Learning_Data_ClampedPinned_Rod_IK/00.-Find Initial Rod Shape/`:
- **Figures**: Contour plots, individual solutions, and combined visualizations (PNG/FIG)
- **Data**: Solution parameters saved as `.mat` files for further analysis

# Examples
```julia
# Using default configuration
success = elliptical_rod_solver()

# Using custom configuration  
config = create_config(xp = 0.5, yp = 0.0, mode = 2.0)
success = elliptical_rod_solver(config)

# For different rod geometries
config = create_config(L = 2.0, EI = 0.5, N = 100)
success = elliptical_rod_solver(config)

# With tilted clamped end
config = create_config(theta = π/4, xp = 0.3, yp = 0.2)
success = elliptical_rod_solver(config)

# Including linear guide constraints
config = create_config(alpha = π/6, lambda = 0.1, xp = 0.4)
success = elliptical_rod_solver(config)
```

# Notes
- Requires MATLAB installation with valid license
- Uses elliptic integral theory for analytical rod shape computation
- Multiple solutions represent different buckling modes
- Results are automatically saved with timestamps for organization
- MATLAB engine errors are handled with troubleshooting guidance

# References
Based on work from University of the Basque Country UPV/EHU (Oscar Altuzarra, 2021)
"""
function elliptical_rod_solver(config::Union{ClampedRodConfig, Nothing} = nothing)
    try
        # Load configuration parameters
        if config === nothing
            config = get_default_config()
            println("✓ Using default configuration\n")
        else
            println("✓ Using provided configuration\n")
        end
        
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
        
        # Extract grid discretization parameters from config
        slope = config.slope           # Controls kr density distribution
        Nkr = Float64(config.Nkr)     # Number of points for kr axis discretization (convert to Float64 for MATLAB)
        Npsi = Float64(config.Npsi)   # Number of points for psi axis discretization (convert to Float64 for MATLAB)
        epsilon = config.epsilon       # Numerical tolerance to avoid singularities
        
        # Extract figure saving parameter from config
        save_figures = config.save_figures  # Whether to save figures
        
        # Print configuration using the utility function
        println("✓ Input parameters loaded from configuration\n")
        print_config(config)


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
        dataset_folder_name = "Rod_Shape"
        println("✓ Generated timestamp for figures: $timestamp_str\n")
        println("✓ Dataset folder name: $dataset_folder_name\n")
        
        # First initialize MATLAB engine
        mat"""
        % Basic MATLAB initialization
        close all ; % clear ; clc;  % Commented out 'clear' to preserve variables between blocks
        fprintf('MATLAB engine initialized\\n');
        """
        
        # Now send the project root path and timestamp to MATLAB
        @mput matlab_project_root timestamp_str dataset_folder_name xp yp mode
        
        mat"""
        % MATLAB path setup with received project root and timestamp
        project_root = matlab_project_root;

        fprintf('Using project root: %s\\n', project_root);

        matlab_base = fullfile(project_root, 'dataset', 'MATLAB code', 'Learning_Data_ClampedPinned_Rod_IK');
        functions_path = fullfile(matlab_base, '00.-Find Initial Rod Shape');
        functions_subpath = fullfile(functions_path, 'Functions');
        
        % Create timestamped DATASET folder for saving data and figures (using consistent timestamp)
        dataset_base = fullfile(matlab_base, '00.-Find Initial Rod Shape', dataset_folder_name);
        if ~exist(dataset_base, 'dir')
            mkdir(dataset_base);
            fprintf('✓ Created dataset folder: %s\\n', dataset_base);
        end
        
        % Create figures subfolder within the dataset folder
        figures_base_folder = fullfile(dataset_base, 'Figures');
        if ~exist(figures_base_folder, 'dir')
            mkdir(figures_base_folder);
            fprintf('✓ Created figures base folder: %s\\n', figures_base_folder);
        end
        
        % Create timestamped subfolder within Figures folder for all plots
        % Format xp value for folder naming (e.g., 0.2 -> X02, 0.5 -> X05)
        xp_scaled = round(xp * 10);
        if xp_scaled < 10
            xp_str = sprintf('X0%d', xp_scaled);
        else
            xp_str = sprintf('X%d', xp_scaled);
        end
        
        % Format yp value for folder naming (e.g., 0.1 -> Y01, 0.0 -> Y00)
        yp_scaled = round(abs(yp) * 10);
        if yp_scaled < 10
            yp_str = sprintf('Y0%d', yp_scaled);
        else
            yp_str = sprintf('Y%d', yp_scaled);
        end
        
        % Create folder name with timestamp, mode, xp and yp identifiers
        figures_folder_name = sprintf('%s_mode_%d_%s_%s', timestamp_str, mode, xp_str, yp_str);
        figures_folder = fullfile(figures_base_folder, figures_folder_name);
        if ~exist(figures_folder, 'dir')
            mkdir(figures_folder);
            fprintf('✓ Created timestamped figures folder: %s\\n', figures_folder_name);
        end
        
        % Display constructed paths for debugging
        fprintf('MATLAB base path: %s\\n', matlab_base);
        fprintf('Functions path: %s\\n', functions_path);
        fprintf('Figures will be saved to: %s\\n', figures_folder);
        
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
        
        fprintf('\\n✓ MATLAB paths configured from project root\\n');
        fprintf('  Base: %s\\n', matlab_base);
        fprintf('  Functions: %s\\n', functions_path);
        fprintf('  Sub-functions: %s\\n', functions_subpath);

        % Verify MATLAB files exist
        if exist('AA_IK_ClampedPinned_Rod_Elliptical.m', 'file') ~= 2
            error('MATLAB script AA_IK_ClampedPinned_Rod_Elliptical.m not found in path');
        end
        if exist('IK_NewRaph.m', 'file') ~= 2
            error('MATLAB function IK_NewRaph.m not found in path');
        end
        """
        
        println("✓ MATLAB paths and functions verified")
        
        # Matlab original code: AA_IK_ClampedPinned_Rod_Elliptical.m
        mat"""
        % Notes(10/08/2025): added code lines
        % --> Send new parameters to Matlab after: 1. Definition of the problem inputs
        % --> Clear any existing structures
        % --> Create input structure as used in AA_IK_ClampedPinned_Rod_Elliptical.m
        % --> Create BVP input structure as used in the MATLAB script
        % --> Store first solution for Julia comparison (added code lines for compare with Julia. 10/08/2025)
        % --> Replaced / symbol to //, along the code
        % --> Comment this code line: %addpath('Functions')
        % --> Save all the plot and Summary of saved figures
        % --> Set project root and timestamp for this MATLAB block
        % --> Setup figures folder for saving plots (using consistent timestamp)
        % --> %save(['CLampedPinnedRod_sol_' num2str(i) '_mode_' num2str(INBVP.mode)_ ],'IN', 'OUT') ;
        % --> %slope = 0.1 ;      % Parameter that controls the density of kr. 
        % --> %epsilon= 1e-9 ;             % To avoid problems at certain values
        %        
        %
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %           UNIVERSITY OF THE BASQUE COUNTRY UPV/EHU
        %
        %           Solution of the Inverse Position Problem of a
        %           Clamped Pinned Rod using elliptic integrals. 
        %
        %                       9 February, 2021
        %
        %           Authors:
        %                       Oscar Altuzarra
        %                       
        %           Contact:
        %                       oscar.altuzarra@ehu.es
        %           
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % DATA
        %           geometry of mechanism and mechanical properties
        %           buckling mode desired for rod
        %           Output position of the end-effector point
        %
        % % PARAMETERS FOR SOLUTION
        %           Sampling grid definition in k and psi
        %
        % % ALGORITHM
        %        1. Definition of the problem inputs
        %        2. Define Grid discretization in k/ kr and psi
        %        3. Solve R, x, y with Elliptic Int k and psi
        %        4. Solve residuals for x and y
        %        5. Use root finding interpolation to get potential solutions
        %        6. Apply Newton scheme on Elliptic Integration to refine them
        %        7. Plot them
        %        8. Use Direct integration BVP to verify them

        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % INIZIALITATION
        close all ; clear ; clc;  % Clear for fresh start of main algorithm
        %addpath('Functions') ;
        
        % Color definition
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
        %%          1. Definition of the problem inputs

        %   IN     Structure in which are geometric and mechanical parameters for
        %          elliptic integrals integration

        IN.L = 1 ;         % Length of the rod [m]
        IN.N = 50 ;        % Number of nodes in which the rod is discretized

        % Mechanical properties of the rod
        IN.EI = 1 ;        % Stiffness of the angular component of deformation

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Position and orientation at the base end

        IN.x0     = 0 ;     % X Coordinate of Clamped-end
        IN.y0     = 0 ;     % Y Coordinate of Clamped-end
        IN.theta  = 0 ;     % Orientation of Clamped-end with X axis
        %%%%%%%%%%%%%%%
        IN.alpha  = 0 ;     % Angle of the linear guide (when not, it is 0) ;
        IN.lambda = 0 ;     % Length of the linear guide (when not, it is 0) ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Position at the extreme end

        IN.xp   =   0.50;     % X coordinate of end-tip [m]
        IN.yp   =   0.00;     % Y coordinate of end-tip [m]

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Buckling Mode desired for the Inverse Kinematics problem

        IN.mode = 2 ;   % Buckling Mode in Elliptic Integrals approach is 
                    % the number of inflection points in the shape of the rod

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   Define structure for BVP solving
        %
        INBVP.L=IN.L;
        INBVP.N=IN.N;
        INBVP.EI=IN.EI;
        INBVP.px_0=IN.x0;
        INBVP.py_0=IN.y0;
        INBVP.theta_0=IN.theta;
        INBVP.px_end=IN.xp;
        INBVP.py_end=IN.yp;
        INBVP.mode=IN.mode;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        # Send parameters to MATLAB and create input structure
        @mput L N EI x0 y0 theta alpha lambda xp yp mode slope Nkr Npsi epsilon save_figures
        
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
        
        fprintf('✓ MATLAB input structures created\\n');
        """
        
        # Execute the main MATLAB elliptical solver algorithm
        # This replicates  AA_IK_ClampedPinned_Rod_Elliptical.m
        @mput matlab_project_root dataset_folder_name timestamp_str xp yp mode
        mat"""
        fprintf('\\n');
        fprintf('⏳ Running elliptical integral solver...\\n');
        fprintf('\\n');
        
        % Set project root and dataset folder name for this MATLAB block
        project_root = matlab_project_root;

        % Setup dataset folder and figures subfolder for saving plots and data (using consistent timestamp)
        matlab_base = fullfile(project_root, 'dataset', 'MATLAB code', 'Learning_Data_ClampedPinned_Rod_IK');
        dataset_base = fullfile(matlab_base, '00.-Find Initial Rod Shape', dataset_folder_name);
        if ~exist(dataset_base, 'dir')
            mkdir(dataset_base);
            fprintf('✓ Created dataset folder: %s\\n', dataset_base);
        end
        
        % Create figures base folder
        figures_base_folder = fullfile(dataset_base, 'Figures');
        if ~exist(figures_base_folder, 'dir')
            mkdir(figures_base_folder);
            fprintf('✓ Created figures base folder: %s\\n', figures_base_folder);
        end
        
        % Create timestamped subfolder within Figures folder for all plots
        % Format xp value for folder naming (e.g., 0.2 -> X02, 0.5 -> X05)
        xp_scaled = round(xp * 10);
        if xp_scaled < 10
            xp_str = sprintf('X0%d', xp_scaled);
        else
            xp_str = sprintf('X%d', xp_scaled);
        end
        
        % Format yp value for folder naming (e.g., 0.1 -> Y01, 0.0 -> Y00)
        yp_scaled = round(abs(yp) * 10);
        if yp_scaled < 10
            yp_str = sprintf('Y0%d', yp_scaled);
        else
            yp_str = sprintf('Y%d', yp_scaled);
        end
        
        % Create folder name with timestamp, mode, xp and yp identifiers
        figures_folder_name = sprintf('%s_mode_%d_%s_%s', timestamp_str, mode, xp_str, yp_str);
        figures_folder = fullfile(figures_base_folder, figures_folder_name);
        if ~exist(figures_folder, 'dir')
            mkdir(figures_folder);
            fprintf('✓ Created timestamped figures folder: %s\\n', figures_folder_name);
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%              2. Define Grid discretization in [k_rel psi]
        %
        %      Discretization of the kr axis ]0 1[ with exponential relationship
        %      so that Nkr distributed points are coarse near 0 and finer near 1
        %
        %slope = 0.1 ;      % Parameter that controls the density of kr.  
                        % Values between 0 and 1 are allowed.
                        % Close to 1 > linear distribution of values from 0 to 1
                        % Close to 0 > finer mesh in values close to 1
                    % Use parameters from Julia configuration
                    % slope already passed from Julia config
                    % Nkr already passed from Julia config
        %%%%%%%%%%%%%%%
        %Nkr = 150 ;         % Number of points in which range of kr is divided
                % Nkr already passed from Julia config
        %%%%%%%%%%%%%%%
        [ A, B ] = p2AB( slope ) ; % Function cretaed to find arguments of exp dist.
        t = linspace(0, 1, Nkr) ;
        kr_range      = A*(exp(B*t)-1) ;    % [0 1]
        %epsilon= 1e-9 ;             % To avoid problems at certain values
                            % epsilon passed from Julia config
        kr_range(1)   = epsilon ;   % No exact 0 > problems with Elliptic Integrals
        kr_range(end) = 1-epsilon ; % No exact 1 > problems with Elliptic Integrals
        %
        %         Linear discretization in psi range with Npsi points
        %%%%%%%%%%%%%%%
        % Npsi = 300 ;         % Number of points in which range of psi is divided
                % Npsi already passed from Julia config
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
        
        % Save the contour plot (conditional)
        if save_figures
            saveas(gcf, fullfile(figures_folder, '01_Contour_Plot_Residuals.png'));
            saveas(gcf, fullfile(figures_folder, '01_Contour_Plot_Residuals.fig'));
        end
        %fprintf('✓ Saved contour plot to figures folder\\n');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Show Potential solutions on Command Window

            fprintf('\\n Unrefined Solutions reached') ;
            fprintf('\\n______________________________________\\n') ;
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
        fprintf('\\n Refined Solutions reached') ;
        fprintf('\\n______________________________________\\n') ;

        for i = 1:nsols
        IN.psi=psisol(i);
        IN.kr=krsol(i);

        %   Newton scheme based on Elliptic Integration solution
        INsolpos(i)=IK_NewtonRaphson_rod( IN );

        psisol(i)=INsolpos(i).psi;
        krsol(i)=INsolpos(i).kr;

        fprintf('\\n Newton-Raphson process Elliptic Successful \\n') ;
        fprintf('\\n______________________________________\\n') ;
        fprintf('\\n Refined Solution = %i', i) ;
        fprintf('\\n______________________________________\\n') ;
        disp(['psi =' num2str(INsolpos(i).psi) ' kr =' num2str(INsolpos(i).kr)] )
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %%        7. Plot Final Refined Solutions
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
        
        % Plot in different Figures each solution
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ii = 1:nsols

            IN.psi=psisol(ii);
            IN.kr=krsol(ii);
            
        [ xi_mod1, yi_mod1 ]         = ClampedPinned_shape( IN ) ;

        [ xL_mod1, yL_mod1, R_mod1,  ~,  ~,  ~,  ~, ~ ] = ClampedPinned_endpoint_pose_force( IN );

        figure ; 
        grid ; axis image ;  hold on ;
        xlabel('x [m]'); ylabel('y [m]'); 
        title(['Solution ',num2str(ii), ' X_P = ',num2str(xL_mod1), ' Y_P = ',num2str(yL_mod1),' Mode = ',num2str(IN.mode)]);
        plot(xi_mod1, yi_mod1, 'Color', color_matrix(ii,:), 'Linewidth', 2) ;

        plot(IN.x0, IN.y0,'ks','MarkerSize',20,'MarkerFaceColor','k') ;
        plot(xL_mod1, yL_mod1,'ks','MarkerSize',20,'MarkerFaceColor','k') ;
        plot(xL_mod1, yL_mod1,'.w','MarkerSize',20,'MarkerFaceColor','w') ;

        %   X Y limits for figure
        xlim([-1 1])
        ylim([-1 1])
        
        % Save individual solution plot to the main figures folder (conditional)
        if save_figures
            filename_base = sprintf('02_Individual_Solution_%d', ii);
            saveas(gcf, fullfile(figures_folder, [filename_base '.png']));
            saveas(gcf, fullfile(figures_folder, [filename_base '.fig']));
        end

        end
        
        fprintf('✓ Saved %d individual solution plots to figures folder\\n', nsols);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot in one Figure all solutions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Ensure figures folder path is available for multiple solutions plot
        if ~exist('figures_folder', 'var')
            matlab_base = fullfile(project_root, 'dataset', 'MATLAB code', 'Learning_Data_ClampedPinned_Rod_IK');
            dataset_base = fullfile(matlab_base, '00.-Find Initial Rod Shape', dataset_folder_name);
            figures_base_folder = fullfile(dataset_base, 'Figures');
            
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
            end
        end
        
        figure ; 
        grid ; axis image ;  hold on ;
        xlabel('x [m]'); ylabel('y [m]'); 
        title('Multiple Solutions');
        for ii = 1:nsols

            IN.psi=psisol(ii);
            IN.kr=krsol(ii);
            
        [ xi_mod1, yi_mod1 ]         = ClampedPinned_shape( IN ) ;

        [ xL_mod1, yL_mod1, R_mod1,  ~,  ~,  ~,  ~, ~ ] = ClampedPinned_endpoint_pose_force( IN );

        plot(xi_mod1, yi_mod1, 'Color', color_matrix(ii,:), 'Linewidth', 2) ;

        plot(IN.x0, IN.y0,'ks','MarkerSize',20,'MarkerFaceColor','k') ;
        plot(xL_mod1, yL_mod1,'ks','MarkerSize',20,'MarkerFaceColor','k') ;
        plot(xL_mod1, yL_mod1,'.w','MarkerSize',20,'MarkerFaceColor','w') ;

        end
        %   X Y limits for figure
        xlim([-1 1])
        ylim([-1 1])
        
        % Save multiple solutions plot (conditional)
        if save_figures
            saveas(gcf, fullfile(figures_folder, '03_Multiple_Solutions.png'));
            saveas(gcf, fullfile(figures_folder, '03_Multiple_Solutions.fig'));
            fprintf('✓ Saved multiple solutions plot to figures folder\\n');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%        8. Apply a BVP Newton scheme to refine them and save files

        %%%%%%%%%% Solve Now with the BVP and save .mat files
        for i = 1:nsols
        fprintf('\\n Refined Solution BVP = %i', i) ;
        fprintf('\\n______________________________________\\n') ;

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
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Save in a *.mat file the solution refined
        % The Structures saved are thos eof BVP, can be used for Stability Analysis

        %save(['CLampedPinnedRod_sol_' num2str(i) '_mode_' num2str(INBVP.mode)_ ],'IN', 'OUT') ;
        
                    % Format xp value for filename (e.g., 0.2 -> X02, 0.5 -> X05, -0.2 -> XN02)
                    xp_scaled = round(INBVP.px_end * 10);
                    if xp_scaled >= 0
                        if xp_scaled < 10
                            xp_str = sprintf('X0%d', xp_scaled);
                        else
                            xp_str = sprintf('X%d', xp_scaled);
                        end
                    else
                        % Handle negative values
                        xp_scaled_abs = abs(xp_scaled);
                        if xp_scaled_abs < 10
                            xp_str = sprintf('XN0%d', xp_scaled_abs);
                        else
                            xp_str = sprintf('XN%d', xp_scaled_abs);
                        end
                    end
                    
                    % Format yp value for filename (e.g., 0.0 -> Y00, 0.3 -> Y03, -0.2 -> YN02)
                    yp_scaled = round(INBVP.py_end * 10);
                    if yp_scaled >= 0
                        if yp_scaled < 10
                            yp_str = sprintf('Y0%d', yp_scaled);
                        else
                            yp_str = sprintf('Y%d', yp_scaled);
                        end
                    else
                        % Handle negative values
                        yp_scaled_abs = abs(yp_scaled);
                        if yp_scaled_abs < 10
                            yp_str = sprintf('YN0%d', yp_scaled_abs);
                        else
                            yp_str = sprintf('YN%d', yp_scaled_abs);
                        end
                    end
                    
                    % Ensure we save in the correct directory (dataset folder)
                    save_filename = ['CLampedPinnedRod_sol_' num2str(i) '_mode_' num2str(INBVP.mode) '_' xp_str '_' yp_str];
                    save_fullpath = fullfile(dataset_base, save_filename);
                    fprintf('\\n');
                    fprintf('Saving solution %d to: %s\\n', i, save_fullpath);
                    save(save_fullpath, 'IN', 'OUT');
                    
                    % Also save to the additional Rotate Clamp directory
                    rotate_clamp_base = fullfile(project_root, 'dataset', 'MATLAB code', 'Learning_Data_ClampedPinned_Rod_IK', 'Rotate Clamp', 'Rotated_Clamp');
                    if ~exist(rotate_clamp_base, 'dir')
                        mkdir(rotate_clamp_base);
                    end
                    save_fullpath_rotate = fullfile(rotate_clamp_base, save_filename);
                    save(save_fullpath_rotate, 'IN', 'OUT');

                    end
        
                    % Summary of saved figures and data
                    fprintf('\\n');
                    fprintf('\\n=== FIGURES SAVED ===\\n');
                    fprintf('All plots saved to: %s\\n', figures_folder);
                    fprintf('\\n=== DATA FILES SAVED ===\\n');
                    fprintf('All .mat files have been saved to TWO locations:\\n');
                    fprintf('1. Primary location: %s\\n', dataset_base);
                    fprintf('2. Rotate Clamp location: %s\\n', rotate_clamp_base);
        """
        


        # Get results back to Julia 
        @mget nsols
        
        println("\n=== MATLAB SOLVER RESULTS ===")
        println("✓ Solution computed successfully")
        println("Number of solutions found: $(nsols)")
        
        return true


    catch e
        if isa(e, MATLAB.MEngineError)
            println("\n ERROR: MATLAB engine not available")
            println("Solutions:")
            println("1. Ensure MATLAB is installed and licensed")
            println("2. Run 'matlab -regserver' in Command Prompt as Administrator")
            println("3. Restart Julia")
            return false
        else
            println("\n ERROR: $(e)")
            return false
        end
    end
end


# Run example only when file is executed directly (not when included)
# NOTE: This auto-execution is disabled when used in pipeline scripts
if !@isdefined(__SOLVER_ALREADY_LOADED__) && abspath(PROGRAM_FILE) == @__FILE__
    # Define a flag to indicate we've loaded this module the first time
    global __SOLVER_ALREADY_LOADED__ = true
    
    println("Starting Elliptical Rod Solver...\n")
    println("File executed: $(basename(@__FILE__))\n")
    
    # Setup project environment (only when run directly)
    project_root = setup_project_environment(activate_env = true, instantiate = false)
    println("✓ Project environment activated: $project_root\n")

    # You can run with default config or create a custom one:
    # Default config:
    #success = elliptical_rod_solver()

    # Or with custom config (example):
    custom_config = create_config(xp = 0.6, yp = 0.0, mode = 2.0)
    success = elliptical_rod_solver(custom_config)

    println()  # Empty line for better readability

    if success
        println("✓ Solver completed successfully!")
    else
        println("✗ Solver encountered errors!")
    end
    
# If you want to see when the module is loaded silently, uncomment the next 3 lines:
# else
#     println(" Module loaded: elliptical_rod_solver function available")
#     println(" (Script execution skipped - use solve_and_prepare_data.jl for custom configs)")
end
