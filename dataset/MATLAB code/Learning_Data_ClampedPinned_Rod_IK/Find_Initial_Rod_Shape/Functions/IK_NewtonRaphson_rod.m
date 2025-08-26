function [ IN ] = IK_NewtonRaphson_rod( IN ) 
% Solution of the Inverse Kinematic Position problem of a ClampedPinned rod


IN.sol = 1 ;

% Maximun (absolute and relative) error tolerance of the residue vector
tol = 1e-8 ;
tol2 = tol^2 ;

itermax=10;

% Initial residue
IN = EllipInteg( IN ) ;
res      = [ IN.xL - IN.xp ;
             IN.yL - IN.yp ] ;
res0     = zeros(size(res)) ;
res_res0 = res - res0 ;


while sum(res.*res) > tol2 && sum(res_res0.*res_res0) > tol2*1e-6
    
    res0 = res ;
    
    % Jacobian matrix of the residue respect to the variables
    J = Jacobian( IN ) ;
    
    % Abort calculation if there is NaN in Jacobian matrix
    if not(sum(sum(abs(J)))>=0)
        fprintf('\n NwRph Aborted: NaN in Jacobian \n') ;
        break ;
    end
    
    % Abort calculation if J is not full rank
    if rank(J)<2
        fprintf('\n NwRph Aborted: Singular Jacobian \n') ;
        break ;
    end
    
    % Increment values of the variables
    delta = -J\res0 ;
    
    % Update variables values
    [IN, delta] = Update_variables( IN, delta ) ;
    
    if IN.sol == 0
        fprintf('\n NwRph Aborted: delta too small \n') ;
        break ;
    end
    
    % New residue
    IN = EllipInteg( IN ) ;
    res      = [ IN.xL - IN.xp ;
                 IN.yL - IN.yp ] ;
    res_res0 = res - res0 ;
    
    subiter=0;
    
    while sum(res.*res) > sum(res0.*res0) && sum(delta.*delta) > tol2*1e-12
        
        subiter=subiter+1;
        if subiter==itermax
            fprintf('\n Newton-Raphson process Elliptic Aborted \n') ;
            fprintf('\n Maximum Number of iterations reached \n') ;
            return
        end
        delta = 0.5*delta ;
        
        IN.psi  = IN.psi - delta(1) ;
        IN.kr   = IN.kr  - delta(2) ;
            
        IN = EllipInteg( IN ) ;
        res      = [ IN.xL - IN.xp ;
                     IN.yL - IN.yp ] ;
        res_res0 = res - res0 ;
        
    end
    
end


if max(abs(res)) > tol || not(sum(abs(res))>=0)
    IN.sol = 0 ;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [IN, delta] = Update_variables( IN, delta ) 

epsilon = 3e-11 ; % The same epsilon used in Jacobian matrix calculation x3

while IN.psi+delta(1)>=2*pi || IN.psi+delta(1)<=0
    delta = 0.5*delta ;
    
    if max(abs(delta)) < 1e-24
        IN.sol = 0 ;
        break ;
    end
end

while sign(IN.kr)~=sign(IN.kr+delta(2)) || ...
        abs(IN.kr+delta(2))>=(1-epsilon) || ...
        abs(IN.kr+delta(2))<=epsilon        
    delta = 0.5*delta ;
    
    if max(abs(delta)) < 1e-24
        IN.sol = 0 ;
        break ;
    end
end

IN.psi  = IN.psi + delta(1) ;
IN.kr   = IN.kr  + delta(2) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ Jacob ] = Jacobian( IN )

epsilon = 1e-11 ;

Jacob = zeros(2) ;

IN.psi = IN.psi + epsilon ;
% [ x_eps, y_eps, ~, ~ ] = ClampedPinned_global( IN ) ;
[  ~,  ~, ~, x_eps, y_eps, ~, ~ ] = ClampedPinned_endpoint_pose_force( IN );

Jacob(:,1) = [ x_eps ;
               y_eps ] - ...
             [ IN.xL ;
               IN.yL ] ;
IN.psi = IN.psi - epsilon ;

IN.kr = IN.kr + epsilon ;
% [ x_eps, y_eps, ~, ~ ] = ClampedPinned_global( IN ) ;
[  ~,  ~, ~, x_eps, y_eps, ~, ~ ] = ClampedPinned_endpoint_pose_force( IN );

Jacob(:,2) = [ x_eps ;
               y_eps ] - ...
             [ IN.xL ;
               IN.yL ] ;

Jacob = Jacob/epsilon ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ IN ] = EllipInteg( IN ) 

% [ x, y, Fx, Fy ] = ClampedPinned_global( IN ) ;
[  ~,  ~,  ~, x, y, Fx, Fy, Mz0 ] = ClampedPinned_endpoint_pose_force( IN );

IN.xL = x ;
IN.yL = y ;
IN.Fx = Fx ;
IN.Fy = Fy ;
IN.Mz0= Mz0;
