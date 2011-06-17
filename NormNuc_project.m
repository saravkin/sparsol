function x = NormNuc_project(m,n,x,tau)

% Minimize ||Z-X||_2  subject to  ||Z||_nuc <= tau

x = reshape(x,m,n);

% L1 projection of SVD
[U,S,V] = svd(x,'econ');
s = diag(S);
s = NormL1_project(s,1,tau);
z = U*diag(s)*V';

% Vectorize result
x = z(:);
