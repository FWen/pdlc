function V = hard_thresh(S,lamda);

n = size(S,1);
V = S;
for k=1:n-1
%     V(1+k:n,k) = shrinkage_Lq(S(1+k:n,k), 0, lamda, 1);
    b = S(1+k:n,k);
    i1 = find(abs(b)<=lamda);
    b(i1) = 0;
    V(1+k:n,k) = b;
    V(k,1+k:n) = V(1+k:n,k)';
end

end
