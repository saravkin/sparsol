function d = NormSR_dual(x,weights,normFun,signPattern)
% Sign-restricted dual gauge wrapper (real numbers)

% Project onto sign pattern (-1 = neg, 0 = any, 1 = pos)
x = x - signPattern .* min(x.*signPattern,0);

% Compute dual norm
d = normFun(x,weights);
