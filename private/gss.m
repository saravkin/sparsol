function [a,b] = gss(f,a,b,eps,N)
%
% Performs golden section search on the function f.
% Assumptions: f is continuous on [a,b]; and
% f has only one minimum in [a,b].
% No more than N function evaluations are done.
% When b-a < eps, the iteration stops.
%
% Example: [a,b] = gss('myfun',0,1,0.01,20)
%
c = (-1+sqrt(5))/2;
x1 = c*a + (1-c)*b;
fx1 = feval(f,x1);
x2 = (1-c)*a + c*b;
fx2 = feval(f,x2);
fprintf('------------------------------------------------------\n');
fprintf(' x1 x2 f(x1) f(x2) b - a\n');
fprintf('------------------------------------------------------\n');
fprintf('%.4e %.4e %.4e %.4e %.4e\n', x1, x2, fx1, fx2, b-a);
for i = 1:N-2
if fx1 < fx2
b = x2;
x2 = x1;
fx2 = fx1;
x1 = c*a + (1-c)*b;
fx1 = feval(f,x1);
else
a = x1;
x1 = x2;
fx1 = fx2;
x2 = (1-c)*a + c*b;
fx2 = feval(f,x2);
end;
fprintf('%.4e %.4e %.4e %.4e %.4e\n', x1, x2, fx1, fx2, b-a);
if (abs(b-a) < eps)
fprintf('succeeded after %d steps\n', i);
return;
end;
end;
fprintf('failed requirements after %d steps\n', N);