%reset(RandStream.getDefaultStream);
ok = 1;
m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));

% put some small guys into x_0
x0(p(k+1:n)) = .005*sign(randn(n-k, 1));


A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.005 * randn(m,1);
tau = norm(x0,1);

%% Subproblem options
options.vapnikEps = 1e-7;
options.lassoOpts.optTol = 1e-8;
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-5*norm(b);
options.primal = 'lsq';

s = 5; % number of outliers
Err = zeros(m, 1); % outliers
pErr = randperm(m);
Err(pErr(1:s)) = .5*randn(s, 1); % outliers are 100 times the size
b = b + Err;

sigma = 1e-2;


%%
options.primal = 'huber';
options.dual   = 'huber';
%options.hparaM = 1e-3;
%options.hparaT = 1e2;
options.hparaM = .005;
options.hparaReg = .005; % this should be roughly the size of the nonzero small coefficients we expect. 
options.exact = 1;


options.exact = 1;
options.rootFinder = 'secant';
[xHuberSecant,info] = gbpdn(A, b, 0, sigma, [], options); 
fprintf('Target tau = %15.7e\n', tau);


expID = 3;
cvx_begin
variable xs(n)
minimize( sum( huber( xs, options.hparaReg))/(2*options.hparaReg) )
subject to
sum(huber(A*xs - b, options.hparaM)/2) <= sigma; % see NOTE
cvx_end

plot(1:n, xHuberSecant - xs)

ok = ok && norm(xHuberSecant - xs, inf) <= 1e-4;




figure(1)
plot(1:n, x0, 1:n, xs - 1, 1:n, xHuberSecant(1:n) + 1);legend('true', 'cvx', 'Secant')



figure(2)
plot(1:m, b - A*x0, 1:m, b - A*xs - 2, 1:m, b - A*xHuberSecant+2);legend('True Outliers', 'cvx res', 'secant res')
