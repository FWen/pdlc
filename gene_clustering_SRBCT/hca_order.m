function Cor = hca_order(CorrelationMat)

n = size(CorrelationMat,1);
D=[];
CorrelationMat = abs(CorrelationMat);
for k=1:n-1
    D = [D, 1-CorrelationMat(k,k+1:end)];
end
tree = linkage(D,'average');
leafOrder = optimalleaforder(tree,D,'Transformation','linear','Criteria','group');

for k=1:n
    for l=1:n
        Cor(k,l) = CorrelationMat(leafOrder(k),leafOrder(l));
    end
end