function y = compHubersTest()

x1 = 100*randn(100, 1);
eps = .1*randn(100, 1);
x2 = x1 + eps;

A = randn(200, 100);
b = randn(200, 1);

%r1 = b - A*x1;
%r2 = b - A*x2; 

%M = abs(5*randn(1));
%T = abs(2*randn(1));
M = 5*abs(randn(200,1));
T = 5*abs(randn(200,1));


[f1 r1 g1] = compHubers(x1, @Aprod, b, M, T);
[f2 r2 g2] = compHubers(x2, @Aprod, b, M, T);


temp = 0.5 * (g1 + g2)'*(eps)/(f2 - f1);

y = abs(1 - temp) < 1e-3;


function z = Aprod(x, mode)

if mode== 1
   z = A*x; 
else if mode == 2
        z = A'*x;
    else
        'wrong mode'
    end
end
        

end

end

