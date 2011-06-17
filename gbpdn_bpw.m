function [x,info] = gbpdn_bpw(A,b,w,options)

if nargin < 4 || isempty(options), options = struct(); end;

% Set norms and projection routine
options.kappa       = @(x)     NormL1_primal(x,w);
options.kappa_polar = @(x)     NormL1_dual(x,w);
options.project     = @(x,tau) NormL1_project(x,w,tau);

[x,info] = gbpdn(A,b,[],0,[],options);

