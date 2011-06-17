function [x,info] = gbpdn_srbp(A,b,s,options)

if nargin < 4 || isempty(options), options = struct(); end;

% Set norms and projection routine
options.kappa       = @(x)     NormSR_primal(x,1,@NormL1_primal,s);
options.kappa_polar = @(x)     NormSR_dual(x,1,@NormL1_dual,s);
options.project     = @(x,tau) NormSR_project(x,1,tau,@NormL1_project,s);

[x,info] = gbpdn(A,b,[],0,[],options);
