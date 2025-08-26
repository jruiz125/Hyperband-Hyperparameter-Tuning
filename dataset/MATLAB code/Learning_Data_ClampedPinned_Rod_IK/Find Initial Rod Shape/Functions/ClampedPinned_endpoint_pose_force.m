function [ xlocal, ylocal, R, x, y, Fx, Fy, Mz0 ] = ClampedPinned_endpoint_pose_force( IN )
% Functions that calculates de position (x,y) of the pinned end of a 
% clamped-pinned rod in the local and global frame, 
% as well as the required load R and the force components of R in global frame (Fx,Fy).
%
% IN :  Structure with the input values
%   IN.L       Rod length [m]
%   IN.EI      Cross-section moment of inertia [m^4]
%   IN.mode    Deformation mode (number of inflection point)
%   IN.psi     Force angle (in local frame) [rad]
%   IN.kr      Elliptic integral parameter (relative value)
%   IN.theta   Angle at the clamped end [rad]
%   IN.x0      Clamped end x-position[m]
%   IN.y0      Clamped end y-position [m]
%   IN.alpha   Angle of the linear guide (when not, it is 0) ;
%   IN.lambda  Length of the linear guide (when not, it is 0) ;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding invalid values of kr.
posNaN = intersect( find(abs(IN.kr)>=1), find(not(abs(IN.kr)>=0)) ) ;
IN.kr(posNaN) = 0.5 ;

% Transformation kr -> k
kmin = abs(cos(0.5*IN.psi)) ;
k    = sign(IN.kr).*kmin + (1-kmin).*IN.kr ;

% Value of phi1 (at clamped end)
aux  = 1./k.*cos(0.5*IN.psi) ;
phi1 = asin(aux) ;

% Elliptic integral of first and second kind, F and E, respectively
[F1, E1] = elliptic12( phi1,             k.^2 );
[F2, E2] = elliptic12( (IN.mode-0.5)*pi, k.^2 ) ;

% Force magnitude R
alpha = F2-F1 ;
R     = alpha.^2*IN.EI/IN.L^2 ;

% Pinned end position in local frame
aux1 = (2*E2-2*E1-F2+F1)./alpha ;
aux2 = -2*cos(phi1)./alpha.*k ;

xlocal = IN.L*(-aux1.*cos(IN.psi)-aux2.*sin(IN.psi)) ;
ylocal = IN.L*(-aux1.*sin(IN.psi)+aux2.*cos(IN.psi)) ;

% Pinned end position in global frame
x = xlocal.*cos(IN.theta) - ylocal.*sin(IN.theta) ;
y = xlocal.*sin(IN.theta) + ylocal.*cos(IN.theta) ;

x = x + IN.x0 + IN.lambda*cos(IN.alpha) ;
y = y + IN.y0 + IN.lambda*sin(IN.alpha) ;

% Force component in global frame
Fx = R.*cos(IN.psi+IN.theta) ;
Fy = R.*sin(IN.psi+IN.theta) ;

% Moment at s=0
Mz0 = 2*k.*sqrt(R.*IN.EI).*cos(phi1);

% NaN number when invalid values of kr
xlocal(posNaN)  = NaN ;
ylocal(posNaN)  = NaN ;
R(posNaN)  = NaN ;
x(posNaN)  = NaN ;
y(posNaN)  = NaN ;
Fx(posNaN) = NaN ;
Fy(posNaN) = NaN ;
Mz0(posNaN) = NaN ;
end