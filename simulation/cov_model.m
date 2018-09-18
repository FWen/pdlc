function X = cov_model(n,model)

X = zeros(n);

if model==1 % Block matrix
    bn = 20;   % block number
    bs = n/bn; % block size
    for ib=1:bn
        X((1:bs)+(ib-1)*bs,(1:bs)+(ib-1)*bs) = 0.8*ones(bs,bs);
    end
    X = X + 0.2*eye(n);
elseif model==2 % Toeplitz matrix
    for k=1:n
        for l=1:n
            X(k,l) = 0.75^abs(k-l);
        end
    end
elseif model==3  % Banded matrix 
    for k=1:n
        for l=1:n
            X(k,l) = (1-abs(k-l)/10) * (abs(k-l)<=10);
        end
    end
    
end