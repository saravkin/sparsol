function [X,info] = cvx_mmv(A,B,sigma,solver);
% Solve using CVX:
%
% Minimize ||X||_1,2 subject to ||AX-B||_F <= sigma
%
% Solver: SeDuMi, SDPT3

[m,n] = size(A);
k     = size(B,2);
info  = struct();

% Start timer
try t0 = toc; catch, tic; t0 = toc; end;

if isreal(A) && isreal(B)
   cvx_begin
      variable X(n,k);
      minimize(sum(norms(X,2,2)));
      subject to
         norm(A*X-B,'fro') <= sigma;
   cvx_end
else
   cvx_begin
      variable X(n,k) complex;
      minimize(sum(norms(X,2,2)));
      subject to
         norm(A*X-B,'fro') <= sigma;
   cvx_end
%   Ar = real(A);
%   Ai = imag(A);
%   Br = real(B);
%   Bi = imag(B);

%   solver_old = cvx_solver;
%   if ~isempty(solver), cvx_solver(solver); end

%   cvx_begin
%      variable X(n,k);
%      variable Y(n,k);
%      minimize(sum(norms([X,Y],2,2)));
%      subject to
%         norm([Ar*X-Ai*Y-Br,Ai*X+Ar*Y-Bi],'fro') <= sigma;
%   cvx_end
%
%   cvx_solver(solver_old);
%
%   X = X + sqrt(-1) * Y;
end

% Stop timer
info.time_solve = toc - t0;
info.time_prepare = 0;
