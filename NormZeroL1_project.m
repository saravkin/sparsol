function x = NormZeroL1_project(x,weights,tau)

idxnz = find(weights > 0);

if isreal(x)
   x(idxnz) = oneProjector(x(idxnz),weights(idxnz),tau);
else
   xa  = abs(x(idxnz));
   idx = xa < eps;
   xc  = oneProjector(xa,weights,tau);
   xc  = xc ./ xa; xc(idx) = 0;
   x(idxnz)   = x(idxnz) .* xc;
end
