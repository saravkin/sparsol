function p = NormNuc_primal(m,n,x)

[U,S,V] = svd(reshape(x,m,n));

p = sum(diag(S));

