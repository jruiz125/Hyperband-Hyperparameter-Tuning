% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%           UNIVERSITY OF THE BASQUE COUNTRY UPV/EHU
%
%           Solution of the Inverse Position Problem of a
%           Clamped Pinned Rod using Runge-Kutta integration from a home
%           position as the Clamping is rotated
%
%                       8 April, 2025
%
%           Authors:
%                       Oscar Altuzarra
%           Contact:
%                       oscar.altuzarra@ehu.es
%           
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
% INIZIALITATION
close all ; clear ; clc;
addpath('Functions') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          1. Loading the initial deformed rod

%   IN     Structure in which are geometric and mechanical parameters
%     IN.L     length of the rod
%     IN.N     number of nodes of the rod for R-K integration
%     IN.EI    Stiffness of the angular component of deformation
%     IN.px_0       X coordinate of clamped origin
%     IN.py_0       Y coordinate of clamped origin 
%     IN.theta_0    Orientation of clamped origin

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load rod structure
load('CLampedPinnedRod_sol_1_mode_2_X02.mat') ;


XP = IN.px_end ;
YP = IN.py_end ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Loaded Rod  
fontsiz = 10 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_rod0 = figure('Position',[60,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ; axis image ;  hold on ;
xlabel('x [m]'); ylabel('y [m]'); 
quiver( [0 0], [0 0], [0.2 0], [0 0.2], 'k', ...
        'linewidth', 1, 'AutoScale', 'off', 'MaxHeadSize', 0.5) ;
text( 0, 0.3, 'Y', 'Units', 'data', 'FontSize', fontsiz, ...
      'HorizontalAlignment', 'center') ;
text( 0.3, 0, 'X', 'Units', 'data', 'FontSize', fontsiz, ...
      'HorizontalAlignment', 'center') ;

title(['Loaded Rod: ', ' X_P = ',num2str(OUT.px(end)), ' Y_P = ',num2str(OUT.py(end))]);

plot(OUT.px,OUT.py,'k-o', 'Linewidth', 2) ; 
% plot(OUT.px(1), OUT.py(1),'ks','MarkerSize',20,'MarkerFaceColor','k') ;
% Rotational actuator
b = 0.01 ;
    th1rot = [ 0.25*pi linspace( 0.5*pi, 1.5*pi, 361 ) 1.75*pi ]  ;
    xrot = b*cos(th1rot) ;
    yrot = b*sin(th1rot) ;
    xrot([1 end]) = 4*sqrt(2)*b*cos(th1rot([1 end])) ;
    yrot([1 end]) = sqrt(2)*b*sin(th1rot([1 end])) ;
    xrot = [ xrot xrot(1) ] + OUT.px(1) ;
    yrot = [ yrot yrot(1) ] + OUT.py(1) ;
    fill(xrot, yrot, 'k', 'EdgeColor', 'none') ;

plot(OUT.px(end), OUT.py(end),'ks','MarkerSize',20,'MarkerFaceColor','k') ;
plot(OUT.px(end), OUT.py(end),'.w','MarkerSize',20,'MarkerFaceColor','w') ;

% Plot end-point force
R =  sqrt(OUT.nx(1)^2+OUT.ny(1)^2) ;
quiver( [OUT.px(end)], [OUT.py(end)], [0.2*OUT.nx(end)/R], [0.2*OUT.ny(end)/R], 'b', 'linewidth', 3, 'MaxHeadSize', 0.5) ;

%   X Y limits for figure
xlim([-0.4 0.9])
ylim([-0.5 0.5])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_rod0_theta_vs_s = figure('Position',[60,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ;   hold on ;
xlabel('s [m]'); ylabel('\theta [rad]'); 
title('Loaded Rod: slope vs arc length');
sz  =   size(OUT.theta);
s   =    linspace(0,IN.L,sz(2));
% Plot 
plot( s , OUT.theta ,'b-o', 'Linewidth', 2) ; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_rod0_curvature_vs_s = figure('Position',[60,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ;   hold on ;
xlabel('s [m]'); ylabel('\kappa [-]'); 
title('Loaded Rod: curvature vs arc length');
sz  =   size(OUT.theta);
s   =    linspace(0,IN.L,sz(2));
% Plot 
plot( s , OUT.m/IN.EI ,'g-o', 'Linewidth', 2) ; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_rod0_curvature_vs_theta = figure('Position',[60,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ;   hold on ;
xlabel('\theta [rad]'); ylabel('\kappa [-]'); 
title('Loaded Rod: curvature vs slope');

% Plot 
plot( OUT.theta , OUT.m/IN.EI ,'b-', 'Linewidth', 2) ; 
plot(OUT.theta(1) , OUT.m(1)/IN.EI,'ko','MarkerSize',10,'MarkerFaceColor','b') ;
plot(OUT.theta(end) , OUT.m(end)/IN.EI,'ks','MarkerSize',10,'MarkerFaceColor','g') ;
%   X Y limits for figure
% xlim([-2*pi 2*pi])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          2. Definition of rotation

% Angle rotated:
theta  = 360*pi/180 ;
% Number of Steps
Ntheta  =   72  ;

% Angle increment at each step:
delta_theta     =    theta / Ntheta  ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          3. Run along

% Prepare Plot for Motion:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1 = figure('Position',[600,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ; axis image ;  hold on ;
xlabel('x [m]'); ylabel('y [m]'); 
quiver( [0 0], [0 0], [0.2 0], [0 0.2], 'k', ...
        'linewidth', 1, 'AutoScale', 'off', 'MaxHeadSize', 0.5) ;
text( 0, 0.3, 'Y', 'Units', 'data', 'FontSize', fontsiz, ...
      'HorizontalAlignment', 'center') ;
text( 0.3, 0, 'X', 'Units', 'data', 'FontSize', fontsiz, ...
      'HorizontalAlignment', 'center') ;

%   X Y limits for figure
xlim([-0.4 0.9])
ylim([-0.5 0.5])

% Prepare Phase Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig2 = figure('Position',[600,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ;  
hold on ;
title('Phase Plot')
xlabel('\theta [rad]','FontSize',14); 
ylabel('\kappa [-]','FontSize',14); 

% Prepare Theta vs Initial curvature Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig3 = figure('Position',[600,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ;  
hold on ;
title('Theta Input vs Initial curvature \kappa_{0} Plot')
xlabel('\theta_{o} [rad]','FontSize',14); 
ylabel('\kappa_{o} [-]','FontSize',14); 

% Prepare Theta vs R Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig4 = figure('Position',[600,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ;  
hold on ;
title('Theta Input vs R Plot')
xlabel('\theta_{o} [rad]','FontSize',14); 
ylabel('R [N]','FontSize',14); 

% Prepare Theta vs Psi Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig5 = figure('Position',[600,60,1000,900]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid ;  
hold on ;
title('Theta Input vs \Psi Plot')
xlabel('\theta_{o} [rad]','FontSize',14); 
ylabel('\Psi [rad]','FontSize',14); 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start Analysis step by step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Solve and save solutions from theta_0 = 0º to 360º
%   There will be N+1 poses for the complete rotation

%thetaStep   = 0 ;
for i = 1:(Ntheta+1)

% Value of theta of local frame wrt Fixed frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    thetaStep   = delta_theta*(i-1) ;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % New coordinates of P in local frame
    IN.px_end   =  XP*cos(thetaStep) + YP*sin(thetaStep); 
    IN.py_end   = -XP*sin(thetaStep) + YP*cos(thetaStep); 
    
    % Solve IK in local frame
    [ IN, OUT, GUESS, RESID ] = IK_fsolve( IN ) ;
    
    if OUT.sol == 1
    % Transform coordinates to fixed frame to plot them
    pX  =   OUT.px.*cos(thetaStep) - OUT.py.*sin(thetaStep) ;
    pY  =   OUT.px.*sin(thetaStep) + OUT.py.*cos(thetaStep) ;

    % nX  =   OUT.nx.*cos(thetaStep) - OUT.ny.*sin(thetaStep) ;
    % nY  =   OUT.nx.*sin(thetaStep) + OUT.ny.*cos(thetaStep) ;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save resuts into Learning Data set in fixed frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save Data set:
        Data(1)   =  pX(1) ;
        Data(2)   =  pY(1) ;
        Data(3)   =  OUT.theta(1) + thetaStep ;
        Data(4)   =  OUT.mode ;
        Data(5)   =  pX(end) ;
        Data(6)   =  pY(end) ;
        Data(7)   =  OUT.m(1) ;
        Data(8)   =  OUT.nx(1); %nX(1) ;
        Data(9)   =  OUT.ny(1); %nY(1) ;
        Data(10)  =  sqrt(OUT.nx(1)^2+OUT.ny(1)^2) ;
        Data(11)  =  atan2(OUT.ny(1),OUT.nx(1)) + thetaStep;

        Datab     =  horzcat(Data,pX,pY,OUT.m,OUT.theta + thetaStep) ;

        if i ==1 
            DataSet  =  Datab ;
        else
            DataSet  =  vertcat(DataSet, Datab) ;
        end
    OUT_last = OUT ;    
    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot last pose: red
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure(fig1)
        title(['\color{red}{Rotation Ended at: }',num2str(thetaStep*180/pi), ' degrees'])
        plot(pX,pY,'-r', 'Linewidth', 2) ; 
        plot(pX(end), pY(end),'rs','MarkerSize',20,'MarkerFaceColor','r') ;
        plot(pX(end), pY(end),'.w','MarkerSize',20,'MarkerFaceColor','w') ;
        % Rotational actuator
            b = 0.01 ;
            th1rot = [ 0.25*pi linspace( 0.5*pi, 1.5*pi, 361 ) 1.75*pi ] ;
            xrot = b*cos(th1rot) ;
            yrot = b*sin(th1rot) ;
            xrot([1 end]) = 4*sqrt(2)*b*cos(th1rot([1 end])) ;
            yrot([1 end]) = sqrt(2)*b*sin(th1rot([1 end])) ;
            xrot = [ xrot xrot(1) ]  ;
            yrot = [ yrot yrot(1) ]  ;
            % Transform coordinates to fixed frame to plot them
            xrotF  =   xrot.*cos(thetaStep) - yrot.*sin(thetaStep) ;
            yrotF  =   xrot.*sin(thetaStep) + yrot.*cos(thetaStep) ;
            fill(xrotF, yrotF, 'r', 'EdgeColor', 'none') ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot last pose: Phase Plot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure(fig2)  
        plot(OUT_last.theta + thetaStep, OUT_last.m/IN.EI,'-r', 'Linewidth', 1) ; 
        plot(OUT_last.theta(1) + thetaStep, OUT_last.m(1)/IN.EI,'ko','MarkerSize',10,'MarkerFaceColor','r') ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Theta_0 vs Initial curvature Plot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure(fig3)
        plot(DataSet(:,3), DataSet(:,7)/IN.EI,'k-o','MarkerSize',10,'MarkerFaceColor','b') ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot last pose: Theta_0 vs Initial curvature Plot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure(fig3)   
        plot(OUT_last.theta(1) + thetaStep, OUT_last.m(1)/IN.EI,'ko','MarkerSize',10,'MarkerFaceColor','r') ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot last pose: Theta_0 vs R Plot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure(fig4)
        plot(DataSet(:,3), sqrt(DataSet(:,8).^2+DataSet(:,9).^2),'k-o','MarkerSize',10,'MarkerFaceColor','b') ;
        plot(OUT_last.theta(1) + thetaStep, sqrt(OUT_last.nx(1)^2+OUT_last.ny(1)^2),'ko','MarkerSize',10,'MarkerFaceColor','r') ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot last pose: Theta_0 vs Psi Plot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure(fig5)
        plot(DataSet(:,3), atan2(DataSet(:,9),DataSet(:,8)) + thetaStep,'k-o','MarkerSize',10,'MarkerFaceColor','b') ;
        plot(OUT_last.theta(1) + thetaStep, atan2(OUT_last.ny(1),OUT_last.nx(1)) + thetaStep,'ko','MarkerSize',10,'MarkerFaceColor','r') ;

    return
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot motion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig1)
plot(pX,pY,'-b', 'Linewidth', 2) ; 
plot(pX(end), pY(end),'ks','MarkerSize',20,'MarkerFaceColor','k') ;
plot(pX(end), pY(end),'.w','MarkerSize',20,'MarkerFaceColor','w') ;

% Rotational actuator
b = 0.01 ;
    th1rot = [ 0.25*pi linspace( 0.5*pi, 1.5*pi, 361 ) 1.75*pi ] ;
    xrot = b*cos(th1rot) ;
    yrot = b*sin(th1rot) ;
    xrot([1 end]) = 4*sqrt(2)*b*cos(th1rot([1 end])) ;
    yrot([1 end]) = sqrt(2)*b*sin(th1rot([1 end])) ;
    xrot = [ xrot xrot(1) ]  ;
    yrot = [ yrot yrot(1) ]  ;
    % Transform coordinates to fixed frame to plot them
    xrotF  =   xrot.*cos(thetaStep) - yrot.*sin(thetaStep) ;
    yrotF  =   xrot.*sin(thetaStep) + yrot.*cos(thetaStep) ;
    fill(xrotF, yrotF, 'k', 'EdgeColor', 'none') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig2)  
plot(OUT.theta + thetaStep, OUT.m/IN.EI,'-', 'Linewidth', 1) ; 
plot(OUT.theta(1) + thetaStep, OUT.m(1)/IN.EI,'ko','MarkerSize',10,'MarkerFaceColor','b') ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theta_0 vs Initial curvature Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig3)
plot(DataSet(:,3), DataSet(:,7)/IN.EI,'k-o','MarkerSize',10,'MarkerFaceColor','b') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theta_0 vs R Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig4)   
plot(DataSet(:,3), DataSet(:,10),'k-o','MarkerSize',10,'MarkerFaceColor','b') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theta_0 vs Psi Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig5)   
plot(DataSet(:,3), DataSet(:,11) ,'k-o','MarkerSize',10,'MarkerFaceColor','b') ;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot last pose: green
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig1)
        title(['\color{green}{Rotation Ended at: }',num2str(thetaStep*180/pi), ' degrees'])
        plot(pX,pY,'-g', 'Linewidth', 2) ; 
        plot(pX(end), pY(end),'gs','MarkerSize',20,'MarkerFaceColor','g') ;
        plot(pX(end), pY(end),'.w','MarkerSize',20,'MarkerFaceColor','w') ;
        % Rotational actuator
            b = 0.01 ;
            th1rot = [ 0.25*pi linspace( 0.5*pi, 1.5*pi, 361 ) 1.75*pi ] ;
            xrot = b*cos(th1rot) ;
            yrot = b*sin(th1rot) ;
            xrot([1 end]) = 4*sqrt(2)*b*cos(th1rot([1 end])) ;
            yrot([1 end]) = sqrt(2)*b*sin(th1rot([1 end])) ;
            xrot = [ xrot xrot(1) ]  ;
            yrot = [ yrot yrot(1) ]  ;
            % Transform coordinates to fixed frame to plot them
            xrotF  =   xrot.*cos(thetaStep) - yrot.*sin(thetaStep) ;
            yrotF  =   xrot.*sin(thetaStep) + yrot.*cos(thetaStep) ;
            fill(xrotF, yrotF, 'g', 'EdgeColor', 'none') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot last pose: Phase Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig2)  
plot(OUT.theta + thetaStep, OUT.m/IN.EI,'-g', 'Linewidth', 1) ; 
plot(OUT.theta(1) + thetaStep, OUT.m(1)/IN.EI,'ko','MarkerSize',10,'MarkerFaceColor','g') ;
plot(DataSet(:,3), DataSet(:,7)/IN.EI,'-k') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot last pose: Theta_0 vs Initial curvature Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig3)
plot(OUT.theta(1) + thetaStep, OUT.m(1)/IN.EI,'ko','MarkerSize',10,'MarkerFaceColor','g') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot last pose: Theta_0 vs R Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig4)   
plot(OUT.theta(1) + thetaStep, sqrt(OUT.nx(1)^2+OUT.ny(1)^2),'ko','MarkerSize',10,'MarkerFaceColor','g') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot last pose: Theta_0 vs Psi Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(fig5)   
plot(OUT.theta(1) + thetaStep, atan2(OUT.ny(1),OUT.nx(1)) + thetaStep,'ko','MarkerSize',10,'MarkerFaceColor','g') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save in a *.mat file the Learning Data Found
save('LearnigData_Rod_ClampedPinned_Rotated_X02_72sols_mode2_revised','DataSet') ;
