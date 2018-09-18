function [X,out] = L1_admm(S,lamda,epsilon,Xtrue);
% Inputs:
%	S: sample covariance
%	lamda: regularization parameter 
%   epsilon: the lower bound for the minimal eigenvalue
%	Xtrue: for debug, for calculation of errors
% Outputs
%	X: the estimation
%	out.e: the error with respect to the true

n = size(S,1);

STD_S = diag(S).^(0.5);
S  = diag(1./STD_S)*S*diag(1./STD_S);   %correlation matrix

X = zeros(n);
if nargin<4
    quiet = 1;
else
    STD_Xtrue = diag(Xtrue).^(0.5);
    Xtrue = diag(1./STD_Xtrue)*Xtrue*diag(1./STD_Xtrue);
    quiet = 0;
end

rho = 1;
max_iter = 50;
ABSTOL   = 1e-6;

W = zeros(n);
out.e=[]; 

for iter = 1:max_iter
    Xm1 = X;
    
    V = X - W/rho;
    for k=1:n-1
        b = V(1+k:n,k);
        V(1+k:n,k) = sign(b) .* max(abs(b)-lamda/rho, 0);
        V(k,1+k:n) = V(1+k:n,k)';
        V(k,k) = 1;
    end
    V(n,n) = 1;
    
    X = 1/(1+rho) * (S + rho*V + W);
     
    [E, U] = eig(X);
    eigV  = diag(U);
    eigV(eigV<epsilon) = epsilon;
    X = E*diag(eigV)*E';
   
    X = (real(X)+real(X)')/2;
    
    W = W - rho*(X - V);
      
    if ~quiet
       % out.e = [out.e,, norm(X-Xm1,'fro')/n];
       % out.e = [out.e, norm(X-Xtrue,'fro')/norm(Xtrue,'fro')];
    end
    
    %terminate when both primal and dual residuals are small
    if (rho*norm(X-Xm1,'fro') < n*ABSTOL && norm(X-V,'fro') < n*ABSTOL) 
        X = diag(STD_S)*X*diag(STD_S);
        return;
    end
    
end
X = diag(STD_S)*X*diag(STD_S);

end

