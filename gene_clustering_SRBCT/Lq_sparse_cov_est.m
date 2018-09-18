function [X,out] = Lq_sparse_cov_est(r,q,epsilon);
% Inputs:
%	r: samples 
%	q: 0<=q<1
%	Xtrue: for debug, for calculation of errors
% Outputs
%	X: the estimation
%	out.e: the error with respect to the true

d = size(r,2);
n = size(r,1);
n2 = round(n/log(n));
n1 = n - n2;

% cross-validation
lamdas = logspace(-3, 0, 15);
for k=1:5
    J  = randperm(n);     % m randomly chosen indices
    S1 = cov(r(J(1:n1),:));
    S2 = cov(r(J(n1+1:end),:));
    parfor l=1:length(lamdas)
        [S_est,~] = Lq_bcd(S1,q,lamdas(l),epsilon);
        FroErr(k,l) = norm(S_est-S2,'fro');
    end
end
[~, mi] = min(sum(FroErr));
lamda = lamdas(mi);

S = cov(r);
[X,out] = Lq_bcd(S,q,lamda,epsilon);
