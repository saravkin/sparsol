function y = hubersTest()

x1 = 100*randn(100, 1);
eps = .1*randn(100, 1);
x2 = x1 + eps;

%M = abs(5*randn(1));
%T = abs(2*randn(1));
M = 5*abs(randn(100,1));
T = 5*abs(randn(100,1));


[f1 g1] = hubers(x1, M, T);
[f2 g2] = hubers(x2, M, T);

temp = 0.5 * (g1 + g2)'*(eps)/(f2 - f1)

y = abs(1 - temp) < 1e-3;

end