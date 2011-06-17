function [x,info] = gbpdn_mmv(A,B,sigma,options)

if nargin < 4 || isempty(options), options = struct(); end;

% Get number of elements per group (number of measurements)
k = size(B,2);

% Set norms and projection routine
weight = getDefaultField(options,'weights',1);
options.kappa       = @(x)     NormL12_primal(k,x,weight);
options.kappa_polar = @(x)     NormL12_dual(k,x,weight);
options.project     = @(x,tau) NormL12_project(k,x,weight,tau);

% Reformulate as BP problem with group structure
if isa(A,'function_handle')
   y = A(B(:,1),2); m = size(B,1); n = length(y);
   A = @(x,mode) blockDiagonalImplicit(A,m,n,k,x,mode);
else
   m = size(A,1); n = size(A,2);
   A = @(x,mode) blockDiagonalExplicit(A,m,n,k,x,mode);
end


% Call solver
[x,info] = gbpdn(A,B(:),[],sigma,[],options);

% Generate output arguments
x = reshape(x,n,k);


function y = blockDiagonalImplicit(A,m,n,g,x,mode)

if mode == 1
   y = zeros(m*g,1);
   for i=1:g
      y(1+(i-1)*m:i*m) = A(x(1+(i-1)*n:i*n),mode);
   end
else
   y = zeros(n*g,1);
   for i=1:g
      y(1+(i-1)*n:i*n) = A(x(1+(i-1)*m:i*m),mode);
   end   
end


function y = blockDiagonalExplicit(A,m,n,g,x,mode)

if mode == 1
   y = A * reshape(x,n,g);
   y = y(:);
else
   x = reshape(x,m,g);
   y = (x' * A)';
   y = y(:);
end

