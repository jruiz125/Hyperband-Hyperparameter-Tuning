function [ IN, OUT, GUESS, RESID ] = IK_NewRaph( IN )
% Shooting method to obtain the solution of the Inverse Kinematic Position
% problem of a sing rod that is clamped at both extremes

tol = 1e-2 ;
tol2 = tol ^2 ;

itermax=10;

OUT = RK4_FullRod( IN ) ;

% OJO no es la misma definición de residuo que antes, igual hay que
% cambiarla. En ese caso hay que cambiar el Jacobiano también.

res = [ OUT.px(end) - IN.px_end ;
          OUT.py(end) - IN.py_end ;
          OUT.m(end) - 0 ] ;
      
GUESS = [ IN.nx0_guess ;
          IN.ny0_guess ;
          IN.m0_guess ] ;
      
RESID = res ;

pp = 1 ;

res0     = zeros(size(res)) ;
res_res0 = res - res0 ;

while sum(res.*res)>tol2 && sum(res_res0.*res_res0)>tol2*1e-9 %max(abs(resid0)) > tol 
%     if pp==itermax
%             fprintf('\n Newton-Raphson process Aborted \n') ;
%             fprintf('\n Maximum Number of principal iterations reached \n') ;
%             fprintf('\n______________________________________\n') ;
%     fprintf('Residue x_end   = %f [m]\n',   OUT.px(end) - IN.px_end ) ;
%     fprintf('Residue y_end   = %f [m]\n',   OUT.py(end) - IN.py_end ) ;
%     fprintf('Residue m_end   = %f [Nm]\n',   OUT.m(end) ) ;
%             return
%     end
    
    res0 = res ;
    
    % Jacobian of the residue vector 
    J = Jacobian( IN, OUT ) ;
    
    if not(sum(sum(abs(J)))>=0)
        fprintf('Jacobian matrix not valid') ;
        break ;
    end
    
    if rank(J)<3
        fprintf('Jacobian matrix rank <3') ;
        break ;
    end
    
    % Increment of the variable to get to the new solutions with less
    % residue 
    delta = -J\res0 ; % \ mldivide: result of Jx=Res0 for x
    
    IN.nx0_guess = IN.nx0_guess + delta(1) ;
    IN.ny0_guess = IN.ny0_guess + delta(2) ;
    IN.m0_guess = IN.m0_guess + delta(3) ;
    
    OUT = RK4_FullRod( IN ) ;
    
    res= [ OUT.px(end) - IN.px_end ;
          OUT.py(end) - IN.py_end ;
          OUT.m(end) - 0 ] ;   
    res_res0 = res - res0 ;
        
    % In case the increment is very high    
    subiter=0;
    
    while sum(res.*res) > sum(res0.*res0) && sum(delta.*delta) > tol2*1e-12 %max(abs(resid)) > max(abs(resid0))
        subiter=subiter+1;
        if subiter==itermax
            fprintf('\n Newton-Raphson process Aborted \n') ;
            fprintf('\n Maximum Number of subiterations reached \n') ;
            fprintf('\n______________________________________\n') ;
    fprintf('Residue x_end   = %f [m]\n',   OUT.px(end) - IN.px_end ) ;
    fprintf('Residue y_end   = %f [m]\n',   OUT.py(end) - IN.py_end ) ;
    fprintf('Residue m_end   = %f [Nm]\n',   OUT.m(end) ) ;
    fprintf('Guess value nx = %f [N]\n',IN.nx0_guess);
    fprintf('Guess value ny = %f [N]\n',IN.ny0_guess);
    fprintf('Guess value m  = %f [Nm]\n',IN.m0_guess);
            return
        end
        
        res0 = res ;
        delta = 0.5*delta ;
        
        IN.nx0_guess = IN.nx0_guess - delta(1) ;
        IN.ny0_guess = IN.ny0_guess - delta(2) ;
        IN.m0_guess = IN.m0_guess - delta(3) ;
       
        OUT = RK4_FullRod( IN ) ;

        res = [ OUT.px(end) - IN.px_end ;
                  OUT.py(end) - IN.py_end ;
                  OUT.m(end) - 0 ] ;
        res_res0 = res - res0 ;
    end
    
    pp = pp + 1 ;
    
    GUESS(:,pp) = [ IN.nx0_guess ;
                    IN.ny0_guess ;
                    IN.m0_guess ] ;
    RESID(:,pp) = res ;
    
end

 if sum(res.*res)<tol2
    OUT.sol = 1 ;
            fprintf('\n Newton-Raphson process Successful \n') ;
            fprintf('\n______________________________________\n') ;
            fprintf('\n Solution reached \n') ;
    fprintf('Theta_0 = %f [m]\n',IN.theta_0*180/pi ) ;        
    fprintf('Residue x_end   = %f [m]\n',   OUT.px(end) - IN.px_end ) ;
    fprintf('Residue y_end   = %f [m]\n',   OUT.py(end) - IN.py_end ) ;
    fprintf('Residue m_end   = %f [Nm]\n',   OUT.m(end) ) ;
    fprintf('Value nx_0 = %f [N]\n',IN.nx0_guess);
    fprintf('Value ny_0 = %f [N]\n',IN.ny0_guess);
    fprintf('Value m_0  = %f [Nm]\n',IN.m0_guess);
    fprintf('Number of principal Iterations   = %f \n',   pp ) ;

    
%      % Elastic energy calculated throught the the Simpson method
%     m2           = OUT.m.^2 ;
%     OUT.Ener = m2(1) + m2(end) + 4*sum(m2(2:2:(IN.N-1))) + ...
%                    2*sum(m2(3:2:(IN.N-2))) ;
%     OUT.Ener = OUT.Ener*IN.L/( (IN.N-1)*6*IN.EI ) ;
%     
%     
%     % Identifying deformation mode (+1 added to make sure that the mode
%     % corresponds with the first buckling mode i.e if the rod is buckling
%     % in the first mode, 1 is stored in OUT.mode 
%     OUT.mode = sum( OUT.m(2:end-1).*OUT.m(1:end-2)<0 ) + 1 ;
%     
%   % Maximun tension (or compression)
%     OUT.sigma = 4/IN.r*abs(OUT.m) + ...
%                     abs( OUT.nx*cos(OUT.theta) + ...
%                          OUT.ny*sin(OUT.theta)) ;
%     OUT.sigma = OUT.sigma/pi/(IN.r)^2/10^6 ;
%      
%     OUT.SIGMAMAX = max(OUT.sigma);
    
else
    OUT.sol = 0 ;
             fprintf('\n Newton-Raphson process Unsuccessful \n') ;
            fprintf('\n______________________________________\n') ;
            fprintf('\n Solution reached \n') ;
    fprintf('Theta_0 = %f [m]\n',IN.theta_0*180/pi ) ;
    fprintf('Residue x_end   = %f [m]\n',   OUT.px(end) - IN.px_end ) ;
    fprintf('Residue y_end   = %f [m]\n',   OUT.py(end) - IN.py_end ) ;
    fprintf('Residue m_end   = %f [Nm]\n',   OUT.m(end) ) ;
    fprintf('Number of principal Iterations   = %f \n',   pp ) ;
    
end

    
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ Jacob ] = Jacobian( IN, OUT )

%%%%%%%%%%%%%%%%%%%%%%
% Determinant factor for convergence and speed !!!!!!!
epsilon = 5e-2 ;
%%%%%%%%%%%%%%%%%%%%%%

Jacob = zeros(3) ;

IN.nx0_guess = IN.nx0_guess + epsilon ;
OUT_eps = RK4_FullRod( IN ) ;

Jacob(:,1) = [ OUT_eps.px(end) -  OUT.px(end) ;
               OUT_eps.py(end) -  OUT.py(end) ;
               OUT_eps.m(end) - 0 ] ;
IN.nx0_guess = IN.nx0_guess - epsilon ;

IN.ny0_guess = IN.ny0_guess + epsilon ;
OUT_eps = RK4_FullRod( IN ) ;

Jacob(:,2) = [ OUT_eps.px(end) -  OUT.px(end) ;
               OUT_eps.py(end) -  OUT.py(end) ;
               OUT_eps.m(end) - 0 ] ;
IN.ny0_guess = IN.ny0_guess - epsilon ;

IN.m0_guess = IN.m0_guess + epsilon ;
OUT_eps = RK4_FullRod( IN ) ;
           
Jacob(:,3) = [ OUT_eps.px(end) -  OUT.px(end) ;
               OUT_eps.py(end) -  OUT.py(end) ;
               OUT_eps.m(end) - 0 ] ;
IN.m0_guess = IN.m0_guess - epsilon ;

Jacob = Jacob*epsilon^-1 ;

