%reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.005 * randn(m,1);
tau = norm(x0,1);

%% Subproblem options
options.vapnikEps = 1e-5;
options.lassoOpts.optTol = 0;
options.solver = 1;     %1 for spg,  2 for pqn
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
options.primal = 'lsq';
options.iterations = 100;
sigma = 1e-8;

%% Exact Newton
options.rootFinder = 'newton';
options.exact = 1;
[xNewton,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Exact secant
options.rootFinder = 'secant';
options.exact = 1;
[xSecant,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Inexact secant
options.rootFinder = 'secant';
options.exact = 2;
[xInexactSecant,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);


figure(1)
plot(1:n, x0, 1:n, xNewton(1:n) -2, 1:n, xInexactSecant(1:n) + 2);legend('True', 'Newton', 'ISecant')



figure(2)
plot(1:m, b - A*x0, 1:m, b - A*xNewton(1:n)+3, 1:m, b - A*xInexactSecant(1:n)-3);legend('True Outliers', 'Newton res', 'ISecant res')



