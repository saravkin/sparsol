function y = hubersCompTest()

x1 = 100*randn(100, 1);
eps = .1*randn(100, 1);
x2 = x1 + eps;

A = randn(200, 100);
b = randn(200, 1);

r1 = b - A*x1;
r2 = b - A*x2; 

%M = abs(5*randn(1));
%T = abs(2*randn(1));
M = 5*abs(randn(200,1));
T = 5*abs(randn(200,1));


[f1 g1] = hubers(r1, M, T);
[f2 g2] = hubers(r2, M, T);

grad1 = -A'*g1;
grad2 = -A'*g2;



temp = 0.5 * (grad1 + grad2)'*(eps)/(f2 - f1);

y = abs(1 - temp) < 1e-3;

end