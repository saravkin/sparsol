function [x,info] = cvx_bp(A,b,solver);
% Solve using CVX:
%
% Minimize ||x||_1 subject to Ax = b
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
         A*x == b;
   cvx_end
else
   cvx_begin
      variable x(n) complex;
      minimize norm(x,1);
      subject to
         A*x == b;
   cvx_end
end

% Stop timer
info.time_solve = toc - t0;
info.time_prepare = 0;