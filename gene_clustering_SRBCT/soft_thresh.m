function V = soft_thresh(S,lamda);

n = size(S,1);
V = S;
for k=1:n-1
    V(1+k:n,k) = sign(S(1+k:n,k)) .* max(abs(S(1+k:n,k))-lamda, 0);
    V(k,1+k:n) = V(1+k:n,k)';
end

end
