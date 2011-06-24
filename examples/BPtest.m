reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.005 * randn(m,1);
tau = norm(x0,1);

%% Subproblem options
options.lassoOpts.optTol = 1e-10;
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
    
%% Exact Newton
options.rootFinder = 'newton';
[xNewton,info] = gbpdn(A, b, 0, 1e-8, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Exact secant
options.rootFinder = 'secant';
[xSecant,info] = gbpdn(A, b, 0, 1e-8, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Inexact secant
options.rootFinder = 'isecant';
[xInexactSecant,info] = gbpdn(A, b, 0, 1e-3, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);
