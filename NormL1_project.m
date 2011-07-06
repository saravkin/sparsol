function x = NormL1_project(x,weights,tau, eps)

if isreal(x)
   x = oneProjectorVap(x,weights,tau,eps);
else
   xa  = abs(x);
   idx = xa < eps;
   xc  = oneProjector(xa,weights,tau);
   xc  = xc ./ xa; xc(idx) = 0;
   x   = x .* xc;
end
