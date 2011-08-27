function [ok] = HuberSecantTestBP(drawplot)

ok = 1;
reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.005 * randn(m,1);
tau = norm(x0,1);

%% Subproblem options
options.vapnikEps = 0;
options.lassoOpts.optTol = 1e-10;
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-10*norm(b);
options.primal = 'lsq';

sigma = 0;


%%
options.primal = 'huber';
options.exact = 1;
options.rootFinder = 'newton';
[xNewton,info] = gbpdn(A, b, [], sigma, [], options);
fprintf('Target tau = %15.7e\n', tau);


options.exact = 2;
[xISecant,info] = gbpdn(A, b, [], sigma, [], options);
fprintf('Target tau = %15.7e\n', tau);

if(drawplot)
    
    figure(1)
    plot(1:n, x0, 1:n, xNewton(1:n) -2, 1:n, xISecant(1:n) + 2);legend('true', 'Newton', 'ISecant')
    
    
    
    figure(2)
    plot(1:m, b - A*x0, 1:m, b - A*xNewton(1:n)+3, 1:m, b - A*xISecant(1:n)-3);legend('True Outliers', 'Huber Newton residuals', 'Huber Isecant Residuals')
end
ok = ok && norm(xISecant - xNewton, inf) < 1e-4;
end