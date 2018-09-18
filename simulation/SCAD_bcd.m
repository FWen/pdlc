function [X,out] = SCAD_bcd(S,lamda,epsilon,Xtrue);
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
S = diag(1./STD_S)*S*diag(1./STD_S); % correlation matrix
X = sign(S) .* max(abs(S)-lamda, 0); % initialization

if nargin<4
    quiet = 1;
else
    STD_Xtrue = diag(Xtrue).^(0.5);
    Xtrue = diag(1./STD_Xtrue)*Xtrue*diag(1./STD_Xtrue);
    quiet = 0;
end

rho = 2;
max_iter = 500;
ABSTOL   = 1e-6;

V1 = zeros(n); 
V2 = zeros(n); 
ck = 1e-1;
dk = 1e-1;

out.e=[]; 

for iter = 1:max_iter
    Xm1 = X;
    rho = rho*1.085;
    
    % V1 subproblem
    Z1 = (rho*X+ck*V1)/(rho+ck);
    for k=1:n-1
        V1(1+k:n,k) = shrinkage_SCAD(Z1(1+k:n,k), lamda/(rho+ck));
        V1(k,1+k:n) = V1(1+k:n,k)';
        V1(k,k) = 1;
    end
    V1(n,n) = 1;
    
    % V2 subproblem
    Z2 = (rho*X+dk*V2)/(rho+dk);
    [E, U] = eig(Z2);
    eigV2 = diag(U);
    eigV2(eigV2<epsilon) = epsilon;
    V2 = E*diag(eigV2)*E';
    
    % X subproblem
    X = 1/(1+2*rho) * (S + rho*(V1 + V2));

    X = (real(X)+real(X)')/2;
    
    if ~quiet
        % out.e = [out.e, norm(X-Xm1,'fro')];
        % out.e  = [out.e, norm(X-Xtrue,'fro')/norm(Xtrue,'fro')];
    end
    
    %terminate when both primal and dual residuals are small
    if (norm(X-Xm1,'fro') < n*ABSTOL) 
        X = diag(STD_S)*X*diag(STD_S);
        return;
    end
    
end
X = diag(STD_S)*X*diag(STD_S);

end


