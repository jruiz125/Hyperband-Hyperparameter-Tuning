function [ A, B ] = p2AB( k )
% Function that calculate  A, B constants of the exponential funcion 
% f(x) = A*(exp(B*x)-1) that is defined within x=0 and x=1 and that fulfill 
% the following conditions :
% f(0) = 0 ;
% f(1) = 1 ;
% df(1)/dx = k ;


% Tolerance error
tol = 1e-6 ;


% Initial range for B (always negative)
B = linspace(-100,0,1e3+1) ;


% Initial residue
res = exp(B).*(B-k)+k ;


% Initial range within the solution is found
aux = res(1:end-1).*res(2:end) ;
ii = find(aux<0) ;
B1 = B(ii) ;        res1 = res(ii) ;
B2 = B(ii+1) ;      res2 = res(ii+1) ;


% Secant method to obtain the B parameter that fulfill the maximum
% tolerance error allowed
while abs(res2) > tol
    
    B3 = B2 - res2/(res2-res1)*(B2-B1) ;       
    res3 = exp(B3).*(B3-k)+k ;
    
    B1 = B2 ;
    B2 = B3 ;
    
    res1 = res2 ;
    res2 = res3 ;   
    
end
B = B2 ;


% Direct Calculation of A parameter
A = 1/(exp(B)-1) ;

end