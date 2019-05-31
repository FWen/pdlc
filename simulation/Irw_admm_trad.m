function [Xcov,out] = Irw_admm_trad(S,lamda,epsilon,Xtrue);
% Inputs:
%	S: sample covariance
%	lamda: regularization parameter 
%   epsilon: the lower bound for the minimal eigenvalue
%	Xtrue: for debug, for calculation of errors
% Outputs
%	X: the estimation
%	out.e: the error with respect to the true

n = size(S,1);
rho = 1;

STD_S = diag(S).^(0.5);
S = diag(1./STD_S)*S*diag(1./STD_S); %correlation matrix
% X = sign(S) .* max(abs(S)-lamda, 0); %initialization
X = zeros(size(S));
if nargin<4
    quiet = 1;
else
    STD_Xtrue = diag(Xtrue).^(0.5);
    Xtrue = diag(1./STD_Xtrue)*Xtrue*diag(1./STD_Xtrue);
    quiet = 0;
end

ABSTOL = 1e-6;
U = zeros(n);
max_iter2 = 50;
max_iter1 = 10;
out.e=[]; 

for j=1:max_iter1
    if j==1
        W = ones(size(S));
    else
        W = 1./(abs(X)+1e-6);
    end
    
    for iter = 1:max_iter2
        Xm1 = X;

        Z = X - U/rho;
        Z = sign(Z) .* max(abs(Z)-lamda*W/rho, 0);
        Z = Z - diag(diag(Z)) + eye(n);

        X = 1/(1+rho) * (S + rho*Z + U);

        [E, V] = eig(X);
        eigV  = diag(V);
        eigV((eigV<epsilon)) = epsilon;
        X = E*diag(eigV)*E';

        X = (real(X)+real(X)')/2;

        U = U - rho*(X - Z);

        if ~quiet
           % out.e = [out.e, norm(X-Xm1,'fro')/n];
           % out.e  = [out.e norm(X-Xtrue,'fro')/norm(Xtrue,'fro')];
        end

        %terminate when both primal and dual residuals are small
        if (rho*norm(X-Xm1,'fro') < n*ABSTOL && norm(X-Z,'fro') < n*ABSTOL) 
            if j==max_iter1
                Xcov = diag(STD_S)*X*diag(STD_S);
                return;                
            else
                break;
            end
        end

    end
end

Xcov = diag(STD_S)*X*diag(STD_S);

end

