function d = NormNuc_dual(m,n,x)

[U,S,V] = svd(reshape(x,m,n));

d = max(diag(S));

