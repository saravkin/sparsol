function p = NormSR_primal(x,weights,normFun,signPattern)
% Sign-restricted primal gauge wrapper (real numbers)

% Infinity when any sign restrictions violated
if any(x.*signPattern < 0)
   p = Inf;
else
   p = normFun(x,weights);
end;

