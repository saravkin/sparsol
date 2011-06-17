function [x,info] = cvx_srbpdn(A,b,s,sigma);
% Solve sign-restricted BPDN using CVX:
%
% Minimize ||x||_1 subject to Ax = b, s.*x >= 0
%
% Solver: SeDuMi, SDPT3

[m,n] = size(A);
info  = struct();

idxPos  = find(s > 0);
idxNeg  = find(s < 0);
idxFree = find(s == 0);

% Start timer
try t0 = toc; catch, tic; t0 = toc; end;

if isreal(A) && isreal(b)
   cvx_begin
      variable x(n);
      minimize ones(1,length(idxPos))*x(idxPos) - ...
               ones(1,length(idxNeg))*x(idxNeg) + ...
               norm(x(idxFree),1);
      subject to
         norm(A*x-b,2) <= sigma;
         x(idxPos) >= 0;
         x(idxNeg) <= 0;
   cvx_end
else
   error('Sign-restricted BPDN is supported only for real problems');
end

% Stop timer
info.time_solve = toc - t0;
info.time_prepare = 0;
