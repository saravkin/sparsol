%reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
%dWeight = 1./(1:n).^1;
%dWeight = dWeight';
weight = 1./(1:n).^1;

contSize = .01;

p = randperm(n); x0 = zeros(n,1); x0(p(1:n)) = weight'.*sign(randn(n,1));
x0 = x0.*(abs(x0) > 5*contSize);

signalNoise = -contSize + 2*contSize*rand(n, 1);
xCont = x0 + signalNoise; % add some random noise

A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*xCont + 0.005 * randn(m,1);

%% Subproblem options
options.vapnikEps = 0;
options.lassoOpts.optTol = 0;
options.solver = 1;     %1 for spg,  2 for pqn
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
options.primal = 'lsq';
sigma = 1e-4;

%% Exact Newton, vapnik parameter = 0
options.rootFinder = 'newton';
options.exact = 1;
[xL1,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Exact Newton, vapnik parameter = 
options.vapnikEps = contSize;
[xVapnik,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

xL1Support = abs(xL1) > options.vapnikEps;
xVapnikSupport = abs(xVapnik) > options.vapnikEps;
AL1 = A*diag(xL1Support); % 
AVapnik = A*diag(xVapnikSupport);


[xL1Final,info] = LSQR(AL1, b); % Find BP sol'n.
[xVapnikFinal,info] = LSQR(AVapnik, b); % Find BP sol'n.



%x0mod = x0.*(abs(x0) > options.vapnikEps);
%xL1mod = xL1(1:n).*(abs(x0) > options.vapnikEps);
%xVapnikmod = xVapnik(1:n).*(abs(x0)>options.vapnikEps);

figure(1)
plot(1:n, x0, 1:n, xL1 -2, 1:n, xVapnik + 2);legend('true', 'l1', 'vapnik')



figure(2)
plot(1:m, b - A*x0, 1:m, b - A*xL1(1:n)+.05, 1:m, b - A*xVapnik(1:n)-.05);legend('True Outliers', 'l1 residuals', 'Vapnik Residuals')


figure(3)
plot(1:n, x0, 1:n, xL1Final -2, 1:n, xVapnikFinal + 2);legend('true', 'l1', 'vapnik')


