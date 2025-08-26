function [ OUT ] = RK4_FullRod ( IN )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function that solves de initial value problem using the method of
% Runge-Kutta of order four.
%   IN     Structure in which are geometric and mechanical parameters
%               IN.L     length of the rod
%               IN.N     number of nodes of the rod for R-K integration
%               IN.EI    Stiffness of the angular component of deformation
%               IN.F_ext Vector with end-tip load
%               IN.m_ext End-tip load moment
%               IN.m0_guess    initial bending moment
%               IN.px_0       X coordinate of clamped origin
%               IN.py_0       Y coordinate of clamped origin 
%               IN.theta_0    Orientation of clamped origin

%   OUT    Structure in which are saved the variables of the solution
%               OUT.px  array (1xIN.N) with the x component of the centroid position in each node of the rod
%               OUT.py  array (1xIN.N) with the y component of the centroid position in each node of the rod
%               OUT.theta Array (1xIN.N) with the theta angle of the cross-section
%               OUT.nx  Array with the component x of the internal force
%               OUT.ny  Array with the component y of the internal force
%               OUT.m   array (1xIN.N) with the bending moment in each node of the rod
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Increment of length for the integration with R-K
ds = IN.L/(IN.N-1) ;

% Vector with the values of the dependent variables at the clamped end
var = [ IN.px_0 ;           % px at the clamped end
        IN.py_0 ;           % py at the clamped end
        IN.theta_0 ;        % angle of the rod at the clamped end
        IN.nx0_guess;       % nx guess value at the clamped end
        IN.ny0_guess;       % ny guess value at the clamped end
        IN.m0_guess ] ; % guess value of the moment at the clamped end


% InizialitationVariables to save the solution
OUT.px = zeros(1,IN.N) ;        % Array with the component x of the centroid position vector
OUT.py = zeros(1,IN.N) ;        % Array with the component y of the centroid position vector
OUT.theta = zeros(1,IN.N) ;     % Array (1xIN.N) with the theta angle of the cross-section
OUT.nx = zeros(1,IN.N) ;        % Array with the component x of the internal force
OUT.ny = zeros(1,IN.N) ;        % Array with the component y of the internal force
OUT.m  = zeros(1,IN.N) ;        % Array with the values of bending moment through the lenght of the rod
OUT.Ener = NaN;                 % Elastic energy of the Rod



% Position and Bending moment at arc-length=0 (clamped end)
OUT.px(1) = var(1) ;   
OUT.py(1) = var(2) ;
OUT.theta(1) = var(3) ;
OUT.nx(1) = var(4) ;   
OUT.ny(1) = var(5) ;
OUT.m(1) = var(6) ;

% "For" loop to integrate the system of differential equations with R-K

for ii = 2:IN.N
    
    % Rugne-Kutta of order four
    k1 = ds *  Right_function( IN, var ) ;
    k2 = ds *  Right_function( IN, var + 0.5*k1 ) ;
    k3 = ds *  Right_function( IN, var + 0.5*k2 ) ;
    k4 = ds *  Right_function( IN, var + k3 ) ;
    
    var = var + (k1 + 2*k2 + 2*k3 + k4)/6 ;
    
    % Save results at each node of the rod
    
    OUT.px(ii) = var(1) ;   % x component of centroid position vector in node ii
    OUT.py(ii) = var(2) ;   % y component of centroid position vector in node ii  
    OUT.theta(ii) = var(3) ;   % theta angle of the cross-section in node ii  
    OUT.nx(ii) = var(4) ;   
    OUT.ny(ii) = var(5) ;
    OUT.m(ii) = var(6) ;   % bending moment in node ii
end

% Elastic energy calculated through the the Simpson method
    m2 = OUT.m.^2 ;
    OUT.Ener = (IN.L)/(IN.N-1)/3 * ( m2(1) + m2(end) + 4*sum(m2(2:2:(IN.N-1))) + ...
                       2*sum(m2(3:2:(IN.N-2))) ) ;
    OUT.Ener = 0.5*OUT.Ener/IN.EI ;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%      Function with the system of differential equations applied

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ func ] = Right_function( IN, var )

% Value of the right part of the system of differential equations
% corresponding to:

% var(1)	px
% var(2)    py
% var(3)    theta
% var(4)	nx
% var(5)    ny
% var(6)    moment

func = [ cos(var(3));
         sin(var(3)) ;
         var(6)/IN.EI ;
         0;
         0;
         var(4)*sin(var(3)) - var(5)*cos(var(3)) ] ;