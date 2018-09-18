clear all; close all; clc;

d = 100; % covariance dimension

SIGMA = cov_model(d, 3);

figure(2);imagesc(SIGMA);title('Original');

Ns = [50, 100, 200, 400, 800];

for iN=1:length(Ns)

    for k=1:10
        [Ns(iN), k]
        
        r = mvnrnd(zeros(d,1), SIGMA, Ns(iN));
        S = r'*r/Ns(iN);
        
        figure(1);subplot(2,3,1); imagesc(S);
        title(['Sample Covariance' ', FE=' num2str(norm(S-SIGMA,'fro'),'% 10.2f') ', SE=' num2str(norm(S-SIGMA),'% 10.2f')]);   
        
        epsilon = 0.01;
        
        %--ADMM algorithm with L1-norm penalty (soft-thresholding)--------
        tic;
        [X1,~] = L1_sparse_cov_est(r,epsilon);
        T_recd(k,1) = toc;
        ErrFro(k,1) = norm(X1-SIGMA,'fro');
        ErrSpe(k,1) = norm(X1-SIGMA);
       
        figure(1);subplot(2,3,2); imagesc(real(X1));
        title(['L1-ADMM' ', FE=' num2str(ErrFro(k,1),'% 10.2f') ', SE=' num2str(ErrSpe(k,1),'% 10.2f')]);
        
        
        %--Proximal BCD algorithm with different penalties---------------
        qs = [0.5, 0]; % for L_(0.5)-, and hard-thresholding penalties
        for iq=1:length(qs)
            tic;
            [X,~] = Lq_sparse_cov_est(r,qs(iq),epsilon);
            T_recd(k,iq+1) = toc;
            ErrFro(k,iq+1) = norm(X-SIGMA,'fro');
            ErrSpe(k,iq+1) = norm(X-SIGMA);
            
            figure(1);subplot(2,3,iq+2); imagesc(X);
            title(['Lq-BCD, (q=' num2str(qs(iq),'%10.1f') '), FE=' num2str(ErrFro(k,iq),'% 10.2f') ', SE=' num2str(ErrSpe(k,iq),'% 10.2f')]);
        end
        
        
        % --Proximal BCD algorithm with SCAD-thresholding
        tic;
        [X,~] = SCAD_sparse_cov_est(r,epsilon);
        T_recd(k,4) = toc;
        ErrFro(k,4) = norm(X-SIGMA,'fro');
        ErrSpe(k,4) = norm(X-SIGMA);
            
        figure(1);subplot(2,3,5); imagesc(X);
        title(['SCAD-BCD', ', FE=' num2str(ErrFro(k,4),'% 10.2f') ', SE=' num2str(ErrSpe(k,4),'% 10.2f')]);


        %--Proposed iteratively reweighted ADMM algorithm
        tic
        [X,~] = Irw_sparse_cov_est(r,epsilon);
        T_recd(k,5) = toc;
        ErrFro(k,5) = norm(X-SIGMA,'fro');
        ErrSpe(k,5) = norm(X-SIGMA);
        
        figure(1);subplot(2,3,6); imagesc(X);
        title(['IRW-ADMM', ', FE=' num2str(ErrFro(k,5),'% 10.2f') ', SE=' num2str(ErrSpe(k,5),'% 10.2f')]);

    end

    AvFE(iN,:) = mean(ErrFro,1);
    AvSE(iN,:) = mean(ErrSpe,1);
    AvT(iN,:)  = mean(T_recd,1);
end

figure(4);subplot(1,2,1);
plot(Ns,AvFE(:,1),'-',Ns,AvFE(:,2),'--+',Ns,AvFE(:,3),'-.',Ns,AvFE(:,4),':x',Ns,AvFE(:,5),':+','linewidth',1);
legend('L1-ADMM','Lq-BCD (q=0.5)','Hard-BCD','SCAD-BCD','IRW-ADMM','Location','Best');
grid; xlim([Ns(1), Ns(end)]); xlabel('Number of samples (N)'); 
ylabel('Averaged relative error (Frobenius norm)'); 

figure(4);subplot(1,2,2);
plot(Ns,AvSE(:,1),'-',Ns,AvSE(:,2),'--+',Ns,AvSE(:,3),'-.',Ns,AvSE(:,4),':x',Ns,AvSE(:,5),':+','linewidth',1);
ylabel('Averaged relative error (Spectral norm)'); 
xlabel('Number of samples (N)'); grid; xlim([Ns(1), Ns(end)]);

figure(5);
plot(Ns,AvT(:,1),'-',Ns,AvT(:,2),'--+',Ns,AvT(:,3),'-.',Ns,AvT(:,4),':x',Ns,AvT(:,5),':+','linewidth',1);
ylabel('Runtime (second)'); 
xlabel('Number of samples (N)'); grid; xlim([Ns(1), Ns(end)]);
legend('L1-ADMM','Lq-BCD (q=0.5)','Hard-BCD','SCAD-BCD','IRW-ADMM','Location','Best');
