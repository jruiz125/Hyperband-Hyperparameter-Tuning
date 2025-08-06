function [ xi, yi ] = ClampedPinned_shape( IN )
% Functions that calculates the clamped-pinned rod shape (xi,yi),
% discretized in n points
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
%   IN.alpha   Angle of the linear guide (when not, it is 0) 
%   IN.lambda  Length of the linear guide (when not, it is 0)
%
%   IN.N :   Number of point to discretize



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transformation kr -> k
kmin = abs(cos(0.5*IN.psi)) ;
k    = sign(IN.kr).*kmin + (1-kmin).*IN.kr ;

% Range of phi
aux  = 1./k.*cos(0.5*IN.psi) ;
phi  = linspace(asin(aux), (IN.mode-0.5)*pi, IN.N) ;

% Elliptic integral of first and second kind, F and E, respectively
[F, E] = elliptic12( phi, k^2 );

% Rod shape in local frame
alpha = F(end)-F(1);

aux1 = (2*E-2*E(1)-F+F(1))/alpha*IN.L;
aux2 = 2*(cos(phi)-cos(phi(1)))/alpha*IN.L*k;

xi_local = -aux1.*cos(IN.psi)-aux2.*sin(IN.psi);
yi_local = -aux1.*sin(IN.psi)+aux2.*cos(IN.psi);

% Rod shape in global frame
xi = xi_local*cos(IN.theta) - yi_local*sin(IN.theta) ;
yi = xi_local*sin(IN.theta) + yi_local*cos(IN.theta) ;

xi = xi + IN.x0 + IN.lambda*cos(IN.alpha) ;
yi = yi + IN.y0 + IN.lambda*sin(IN.alpha) ;


end
