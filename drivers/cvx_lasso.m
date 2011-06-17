function [x,info] = cvx_lasso(A,b,tau);
% Solve using CVX:
%
% Minimize ||Ax-b||_2 subject to ||x||_1 <= tau
%

[m,n] = size(A);
info  = struct();

% Start timer
try t0 = toc; catch, tic; t0 = toc; end;

if isreal(A) && isreal(b)
   cvx_begin
      variable x(n);
      minimize norm(A*x-b,2);
      subject to
         norm(x,1) <= tau;
   cvx_end
else
   cvx_begin
      variable x(n) complex;
      minimize norm(A*x-b,2);
      subject to
         norm(x,1) <= tau;
   cvx_end
end

% Stop timer
info.time_solve = toc - t0;
info.time_prepare = 0;
