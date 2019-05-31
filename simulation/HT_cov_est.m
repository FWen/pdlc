function X = HT_cov_est(r);

d = size(r,2);
n = size(r,1);
n2 = round(n/log(n));
n1 = n - n2;

% cross-validation
lamdas = logspace(-4, 0, 30);
for k=1:5
    J  = randperm(n);     % m randomly chosen indices
    S1 = cov(r(J(1:n1),:));
    S2 = cov(r(J(n1+1:end),:));
    for l=1:length(lamdas)
        S_est = hard_thresh(S1,lamdas(l));
        FroErr(k,l) = norm(S_est-S2,'fro');
    end
end
[m mi] = min(sum(FroErr));
lamda = lamdas(mi);
% figure(3);subplot(211);plot(lamdas,sum(FroErr),'-*');set(gca,'xscale','log');

S = cov(r);
X = hard_thresh(S,lamda);
