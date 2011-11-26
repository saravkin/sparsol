function x = Huber_project(x,weights,tau, eps)

if isreal(x)
   x = oneProjectorHuber(x,weights,tau,eps);
else
   xa  = abs(x);
   idx = xa < eps;
   xc  = oneProjectorHuber(xa,weights,tau);
   xc  = xc ./ xa; xc(idx) = 0;
   x   = x .* xc;
end
