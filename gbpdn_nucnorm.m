function [x,info] = gbpdn_nucnorm(A,b,m,n,sigma,options)
%  Minimize ||X||_nuc  subject to  ||A*vec(X)-b||_F <= sigma
%
% with X an m-by-n matrix. There are two possible combinations
% for A and b depending on whether X is regarded as a matrix
% of a vector.
%
% 1) When X is a matrix we have A is a k-by-m matrix and b is a
%    k-by-n matrix;
% 2) In the more general case where X is vectorized, A is a
%    k-by-mn matrix, and b is a k-by-1 vector.

if nargin < 6 || isempty(options), options = struct(); end;

if size(b,2) == n
   % Mode 1 - reformulate A
   if isa(A,'function_handle')
      A = @(x,mode) blockDiagonalImplicit(A,size(b,1),m*n,size(b,2),x,mode);
   else
      A = @(x,mode) blockDiagonalExplicit(A,size(b,1),m*n,size(b,2),x,mode);
   end
else
   % Mode 2 - do nothing
end

% Set norms and projection routine
options.kappa       = @(x)     NormNuc_primal(m,n,x);
options.kappa_polar = @(x)     NormNuc_dual(m,n,x);
options.project     = @(x,tau) NormNuc_project(m,n,x,tau);

% Call solver
[x,info] = gbpdn(A,b,[],sigma,[],options);

% Generate output arguments
x = reshape(x,m,n);


function y = blockDiagonalImplicit(A,m,n,k,x,mode)

if mode == 1
   y = zeros(m*k,1);
   for i=1:k
      y(1+(i-1)*m:i*m) = A(x(1+(i-1)*n:i*n),mode);
   end
else
   y = zeros(n*k,1);
   for i=1:k
      y(1+(i-1)*n:i*n) = A(x(1+(i-1)*m:i*m),mode);
   end   
end


function y = blockDiagonalExplicit(A,m,n,k,x,mode)

if mode == 1
   y = A * reshape(x,n,k);
   y = y(:);
else
   x = reshape(x,m,k);
   y = (x' * A)';
   y = y(:);
end

