function [x,info] = cvx_bpdn(A,b,sigma,solver);
% Solve using CVX:
%
% Minimize ||x||_1 subject to ||Ax-b||_2 <= sigma
%
% Solver: SeDuMi, SDPT3

[m,n] = size(A);
info  = struct();

% Start timer
try t0 = toc; catch, tic; t0 = toc; end;

if isreal(A) && isreal(b)
   cvx_begin
      variable x(n);
      minimize norm(x,1);
      subject to
         norm(A*x-b,2) <= sigma;
   cvx_end
else
   cvx_begin
      variable x(n) complex;
      minimize norm(x,1);
      subject to
         norm(A*x-b,2) <= sigma;
   cvx_end
end

% Stop timer
info.time_solve = toc - t0;
info.time_prepare = 0;