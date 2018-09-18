function [X,out] = SCAD_sparse_cov_est(r,epsilon);
% Inputs:
%	r: samples 
%	Xtrue: for debug, for calculation of errors
% Outputs
%	X: the estimation

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
        [S_est,~]  = SCAD_bcd(S1,lamdas(l),epsilon);
        FroErr(k,l) = norm(S_est-S2,'fro');
    end
end
[~, mi] = min(sum(FroErr));
lamda = lamdas(mi);

S = cov(r);
[X,out] = SCAD_bcd(S,lamda,epsilon);
