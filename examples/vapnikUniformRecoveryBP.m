reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
%dWeight = 1./(1:n).^1;
%dWeight = dWeight';

eps = k/(2*n);
residSize = 0;

p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
x0 = x0 + eps*sign(randn(n, 1)); % add some random noise

A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
mult = diag(10*rand(m, 1));
A = mult*A;

b  = A*x0 + residSize * randn(m,1);


%% Subproblem options
options.vapnikEps = 0;
options.lassoOpts.optTol = 0;
options.solver = 1;     %1 for spg,  2 for pqn
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
options.primal = 'lsq';
tau = k;

%% Exact Newton, vapnik parameter = 0
options.rootFinder = 'secant';
options.exact = 2;
[xL1,info] = gbpdn(A, b, [], [], [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Exact Newton, vapnik parameter = 
options.vapnikEps = eps;
[xVapnik,info] = gbpdn(A, b, [], [], [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);


%x0mod = x0.*(abs(x0) > options.vapnikEps);
%xL1mod = xL1(1:n).*(abs(x0) > options.vapnikEps);
%xVapnikmod = xVapnik(1:n).*(abs(x0)>options.vapnikEps);
x0mod = x0;
xVapnikmod = xVapnik;
xL1mod = xL1;


figure(1)
plot(1:n, x0mod, 1:n, xL1mod -2, 1:n, xVapnikmod + 2);legend('true', 'l1', 'vapnik')
hold on;
p = plot(1:n, eps*ones(n, 1), 1:n, -eps*ones(n, 1),...
    1:n, 2 + eps*ones(n, 1),  1:n, 2 - eps*ones(n, 1), ...
    1:n, -2 - eps*ones(n, 1), 1:n, -2 + eps*ones(n, 1));
set(p,'Color','black','LineWidth',1)
hold off;

figure(2)
plot(1:m, b - A*x0, 1:m, b - A*xL1(1:n)+.05, 1:m, b - A*xVapnik(1:n)-.05);legend('True Outliers', 'l1 residuals', 'Vapnik Residuals')



