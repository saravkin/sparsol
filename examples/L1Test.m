%reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';

km = 10;
pm = randperm(m); v0 = zeros(m,1); v0(pm(1:km)) = sign(randn(km,1));
b  = A*x0 + 0.05 * v0;
tau = norm(x0,1);

%% Subproblem options
options.vapnikEps = 0;
options.lassoOpts.optTol = 0;
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
options.primal = 'lsq';
options.exact  = 1;


sigma = 1e-2;


%%
options.primal = 'lsq';
options.rootFinder = 'newton';
[xL2, info] = gbpdn(A, b, 0, sigma, [], options); 
fprintf('Target tau = %15.7e\n', tau);

%%
options.primal = 'l1';
%options.hparaM = 1e-3;
%options.hparaT = 1e2;
options.rootFinder = 'secant';
options.exact = 1;
options.lassoOpts.optTol = 1e-8;
options.L1Eps = 1e-6;
sigma = 0.5;
[xL1,info] = gbpdn(A, b, 0, sigma, [], options); 
fprintf('Target tau = %15.7e\n', tau);



figure(1)
plot(1:n, x0, 1:n, xL2(1:n) -2, 1:n, xL1(1:n) + 2);legend('true', 'l2', 'l1')



figure(2)
plot(1:m, b - A*x0, 1:m, b - A*xL2(1:n)+0.1, 1:m, b - A*xL1(1:n)-0.1);legend('True Outliers', 'l2 residuals', 'L1 Residuals')


