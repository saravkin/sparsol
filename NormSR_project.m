function x = NormSR_project(x,weights,tau,projectFun,signPattern)
% Sign-restricted projection wrapper (real numbers)

% Project onto sign pattern (-1 = neg, 0 = any, 1 = pos)
x = x - signPattern .* min(x.*signPattern,0);

% Project second stage
x = projectFun(x,weights,tau);
