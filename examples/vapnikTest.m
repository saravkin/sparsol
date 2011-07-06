%reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
dWeight = 1./(1:n).^1;
dWeight = dWeight';

p = randperm(n); x0 = zeros(n,1); x0(p(1:n)) = sign(randn(n,1));
x0(p(1:n)) = x0.*dWeight;

A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.01 * randn(m,1);
tau = norm(x0,1);

%% Subproblem options
options.vapnikEps = 0;
options.lassoOpts.optTol = 0;
options.solver = 1;     %1 for spg,  2 for pqn
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
options.primal = 'lsq';
sigma = 1e-6;

%% Exact Newton, vapnik parameter = 0
options.rootFinder = 'newton';
options.exact = 1;
[xL1,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Exact Newton, vapnik parameter = 
options.vapnikEps = 0.01;
[xVapnik,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);


x0mod = x0.*(abs(x0) > options.vapnikEps);
xL1mod = xL1(1:n).*(abs(x0) > options.vapnikEps);
xVapnikmod = xVapnik(1:n).*(abs(x0)>options.vapnikEps);

figure(1)
plot(1:n, x0mod, 1:n, xL1mod -2, 1:n, xVapnikmod + 2);legend('true', 'l1', 'vapnik')



figure(2)
plot(1:m, b - A*x0, 1:m, b - A*xL1(1:n)+3, 1:m, b - A*xVapnik(1:n)-3);legend('True Outliers', 'l1 residuals', 'Vapnik Residuals')



