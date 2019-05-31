clear all;  clc;%close all;

d = 100; % covariance dimension

SIGMA = cov_model(d, 3);

% figure(1); subplot(121); imagesc(SIGMA);title('Original');

Ns = [50, 100, 200, 400, 800];

for iN=1:length(Ns)

    for k=1:20
        [Ns(iN), k]
        
        r = mvnrnd(zeros(d,1), SIGMA, Ns(iN));
        S = cov(r);% 
        STD_S = diag(S).^(0.5); %std
        
        
        epsilon = 1e-3; % lower bound for the eigenvalue
        
        %--ADMM algorithm with L1-norm penalty (soft-thresholding)--------
        tic;
        [X,~] = L1_sparse_cov_est(r,epsilon);
        T_recd(k,1) = toc;
        ErrFro(k,1) = norm(X-SIGMA,'fro');
        ErrSpe(k,1) = norm(X-SIGMA);
        Eigs(k,1,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
        Eigs_cov(k,1,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
        
        figure(2);subplot(3,3,1); imagesc(real(X));
        title(['L1-ADMM' ', FE=' num2str(ErrFro(k,1),'% 10.2f') ', SE=' num2str(ErrSpe(k,1),'% 10.2f')]);
        
        
        %--Proximal BCD algorithm with different penalties---------------
        qs = [0.5, 0]; % for L_(0.5)-, and hard-thresholding penalties
        for iq=1:length(qs)
            tic;
            [X,~] = Lq_sparse_cov_est(r,qs(iq),epsilon);
            T_recd(k,iq+1) = toc;
            ErrFro(k,iq+1) = norm(X-SIGMA,'fro');
            ErrSpe(k,iq+1) = norm(X-SIGMA);
            Eigs(k,iq+1,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
            Eigs_cov(k,iq+1,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
            figure(2);subplot(3,3,iq+1); imagesc(X);
            title(['Lq-BCD, (q=' num2str(qs(iq),'%10.1f') '), FE=' num2str(ErrFro(k,iq+1),'% 10.2f') ', SE=' num2str(ErrSpe(k,iq+1),'% 10.2f')]);
        end
        
        
        %--Proximal BCD algorithm with SCAD-thresholding
        tic;
        [X,~] = SCAD_sparse_cov_est(r,epsilon);
        T_recd(k,4) = toc;
        ErrFro(k,4) = norm(X-SIGMA,'fro');
        ErrSpe(k,4) = norm(X-SIGMA);
        Eigs(k,4,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
        Eigs_cov(k,4,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
        figure(2);subplot(3,3,4); imagesc(X);
        title(['SCAD-BCD', ', FE=' num2str(ErrFro(k,4),'% 10.2f') ', SE=' num2str(ErrSpe(k,4),'% 10.2f')]);


        %--Proposed iteratively reweighted ADMM algorithm
        tic
        [X,~] = Irw_sparse_cov_est(r,epsilon);
        T_recd(k,5) = toc;
        ErrFro(k,5) = norm(X-SIGMA,'fro');
        ErrSpe(k,5) = norm(X-SIGMA);
        Eigs(k,5,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
        Eigs_cov(k,5,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
        figure(2);subplot(3,3,5); imagesc(X);
        title(['IRW-ADMM', ', FE=' num2str(ErrFro(k,5),'% 10.2f') ', SE=' num2str(ErrSpe(k,5),'% 10.2f')]);
        
        
        %--Traditional iteratively reweighted algorithm using two loops,
        %--with the inner loop solved by ADMM
        tic
        [X,~] = Irw_sparse_cov_est_trad(r,epsilon);
        T_recd(k,6) = toc;
        ErrFro(k,6) = norm(X-SIGMA,'fro');
        ErrSpe(k,6) = norm(X-SIGMA);
        Eigs(k,6,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
        Eigs_cov(k,6,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
        figure(2);subplot(3,3,6); imagesc(X);
        title(['Trad. IRW', ', FE=' num2str(ErrFro(k,6),'% 10.2f') ', SE=' num2str(ErrSpe(k,6),'% 10.2f')]);

        
        %--Soft-thresholding---------------
        tic;
        X = SOT_cov_est(r);
        T_recd(k,7) = toc;
        ErrFro(k,7) = norm(X-SIGMA,'fro');
        ErrSpe(k,7) = norm(X-SIGMA);
        Eigs(k,7,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
        Eigs_cov(k,7,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
        figure(2);subplot(3,3,7); imagesc(X);
        title(['Soft thresholding', ', FE=' num2str(ErrFro(k,7),'% 10.2f') ', SE=' num2str(ErrSpe(k,7),'% 10.2f')]);

        
        %--Hard-thresholding---------------
        tic;
        X = HT_cov_est(r);
        T_recd(k,8) = toc;
        ErrFro(k,8) = norm(X-SIGMA,'fro');
        ErrSpe(k,8) = norm(X-SIGMA);
        Eigs(k,8,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
        Eigs_cov(k,8,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
        figure(2);subplot(3,3,8); imagesc(X);
        title(['Hard thresholding', ', FE=' num2str(ErrFro(k,8),'% 10.2f') ', SE=' num2str(ErrSpe(k,8),'% 10.2f')]);
        

        %--SCAD-thresholding---------------
        tic;
        X = SCADT_cov_est(r);
        T_recd(k,9) = toc;
        ErrFro(k,9) = norm(X-SIGMA,'fro');
        ErrSpe(k,9) = norm(X-SIGMA);
        Eigs(k,9,iN)   = min(real(eig(diag(1./STD_S)*X*diag(1./STD_S)))); %minimal eigen-value of the correaltion matrix
        Eigs_cov(k,9,iN) = min(real(eig(X))); %minimal eigen-value of the covariance matrix
        figure(2);subplot(3,3,9); imagesc(X);
        title(['SCAD thresholding', ', FE=' num2str(ErrFro(k,9),'% 10.2f') ', SE=' num2str(ErrSpe(k,9),'% 10.2f')]);

    end

    AvFE(iN,:) = mean(ErrFro,1);
    AvSE(iN,:) = mean(ErrSpe,1);
    AvT(iN,:)  = mean(T_recd,1);
end

figure(4);subplot(1,2,1);
plot(Ns,AvFE(:,1),'-',Ns,AvFE(:,2),'g--+',Ns,AvFE(:,3),'g-->',Ns,AvFE(:,4),'g--x',Ns,AvFE(:,5),'r--*',...
     Ns,AvFE(:,6),'r--p',Ns,AvFE(:,7),'b--o',Ns,AvFE(:,8),'b--^',Ns,AvFE(:,9),'b--d','linewidth',1);
legend('L1-ADMM','Lq-BCD (q=0.5)','Hard-BCD','SCAD-BCD','IRW-ADMM','IRW-trad','Soft thresh.','Hard thresh.','SCAD thresh.','Location','Best');
grid; xlim([Ns(1), Ns(end)]); xlabel('Number of samples (N)'); 
ylabel('Averaged relative error (Frobenius norm)'); 

figure(4);subplot(1,2,2);
plot(Ns,AvSE(:,1),'-',Ns,AvSE(:,2),'g--+',Ns,AvSE(:,3),'g-->',Ns,AvSE(:,4),'g--x',Ns,AvSE(:,5),'r--*',...
     Ns,AvSE(:,6),'r--p',Ns,AvSE(:,7),'b--o',Ns,AvSE(:,8),'b--^',Ns,AvSE(:,9),'b--d','linewidth',1);
ylabel('Averaged relative error (Spectral norm)'); 
xlabel('Number of samples (N)'); grid; xlim([Ns(1), Ns(end)]);

figure(5);
plot(Ns,AvT(:,1),'-',Ns,AvT(:,2),'g--+',Ns,AvT(:,3),'g-->',Ns,AvT(:,4),'g--x',Ns,AvT(:,5),'r--*',...
     Ns,AvT(:,6),'r--p',Ns,AvT(:,7),'b--o',Ns,AvT(:,8),'b--^',Ns,AvT(:,9),'b--d','linewidth',1);
ylabel('Runtime (second)'); 
xlabel('Number of samples (N)'); grid; xlim([Ns(1), Ns(end)]);
legend('L1-ADMM','Lq-BCD (q=0.5)','Hard-BCD','SCAD-BCD','IRW-ADMM','IRW-trad','Soft thresh.','Hard thresh.','SCAD thresh.','Location','Best');

Eigs
Eigs_cov