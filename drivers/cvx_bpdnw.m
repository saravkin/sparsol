function [x,info] = cvx_bpdnw(A,b,w,sigma);
% Solve using CVX: W = diag(w)
%
% Minimize ||Wx||_1 subject to Ax = b
%
% Solver: SeDuMi, SDPT3

[m,n] = size(A);
info  = struct();

W = spdiags(w,0,length(w),length(w));

% Start timer
try t0 = toc; catch, tic; t0 = toc; end;

if isreal(A) && isreal(b)
   cvx_begin
      variable x(n);
      minimize norm(W*x,1);
      subject to
         norm(A*x - b,2) <= sigma;
   cvx_end
else
   cvx_begin
      variable x(n) complex;
      minimize norm(W*x,1);
      subject to
         norm(A*x - b,2) <= sigma;
   cvx_end
end

% Stop timer
info.time_solve = toc - t0;
info.time_prepare = 0;
