%reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.005 * randn(m,1);
tau = norm(x0,1);

%% Subproblem options
options.vapnikEps = 1e-7;
options.lassoOpts.optTol = 1e-10;
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
options.primal = 'lsq';

s = 5; % number of outliers
Err = zeros(m, 1); % outliers
pErr = randperm(m);
Err(pErr(1:s)) = .5*randn(s, 1); % outliers are 100 times the size
b = b + Err;

sigma = 1e-2;


%%
options.primal = 'huber';
%options.hparaM = 1e-3;
%options.hparaT = 1e2;
options.hparaM = .005;
options.hparaReg = .005; % this should be roughly the size of the nonzero small coefficients we expect. 
options.exact = 1;


options.exact = 1;
options.rootFinder = 'secant';
[xHuberSecant,info] = gbpdn(A, b, 0, sigma, [], options); 
fprintf('Target tau = %15.7e\n', tau);

options.exact = 2;
[xHuberISecant,info] = gbpdn(A, b, 0, sigma, [], options); 
fprintf('Target tau = %15.7e\n', tau);


figure(1)
plot(1:n, x0+ 1, 1:n, xHuberISecant(1:n) - 1);legend('true', 'ISecant')



figure(2)
plot(1:m, b - A*x0 + 2, 1:m, b - A*xHuberISecant(1:n)-2);legend('True Outliers', 'Huber Isecant Residuals')


