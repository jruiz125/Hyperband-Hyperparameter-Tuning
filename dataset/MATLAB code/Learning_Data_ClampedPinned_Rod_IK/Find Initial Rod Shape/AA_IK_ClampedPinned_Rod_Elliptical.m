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
close all ; clear ; clc;
addpath('Functions') ;

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
xlabel('\psi')
ylabel('Kr')
title('\color{blue}{res_X} = 0, \color{red}{res_Y} = 0')

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
%%        7. Plot Final Refined Solutions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot in one Figure all solutions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save in a *.mat file the solution refined
% The Structures saved are thos eof BVP, can be used for Stability Analysis

save(['CLampedPinnedRod_sol_' num2str(i) '_mode_' num2str(INBVP.mode) ],'IN', 'OUT') ;

end
