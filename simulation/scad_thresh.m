function V = scad_thresh(S,lamda);

n = size(S,1);
V = S;
for k=1:n-1
    V(1+k:n,k) = shrinkage_SCAD(S(1+k:n,k), lamda);
    V(k,1+k:n) = V(1+k:n,k)';
end

end
