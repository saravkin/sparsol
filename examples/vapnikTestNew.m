% reset(RandStream.getDefaultStream);
% 
% m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
% %dWeight = 1./(1:n).^1;
% %dWeight = dWeight';
% 
% p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
% x0 = x0 + 0.02*sign(randn(n, 1)); % add some random noise
% 
% A  = randn(m,n); % [Q,R] = qr(A',0);  A = Q';
% b  = A*x0 + 0.005 * randn(m,1);
% tau = norm(x0,1);
% 
% %% Subproblem options
% options.vapnikEps = 0;
% options.lassoOpts.optTol = 0;
% options.solver = 1;     %1 for spg,  2 for pqn
% options.lassoOpts.verbosity = 0;
% options.tolerance = 1e-7*norm(b);
% options.primal = 'lsq';
% sigma = 1e-4;
% tau = 25;
% 
% %% Exact Newton, vapnik parameter = 0
% options.rootFinder = 'newton';
% options.exact = 1;
% %[xL1,info] = gbpdn(A, b, tau, [], [], options); % Find BP sol'n.
% [xL1,info] = gbpdn(A, b, [], sigma, [], options); % Find BP sol'n.
% 
% fprintf('Target tau = %15.7e\n', tau);
% 
% %% Exact Newton, vapnik parameter = 
% options.vapnikEps = 0.02;
% %[xVapnik,info] = gbpdn(A, b, tau - n*options.vapnikEps, [], [], options); % Find BP sol'n.
% [xVapnik,info] = gbpdn(A, b, [], sigma, [], options); % Find BP sol'n.
% 
% 
% fprintf('Target tau = %15.7e\n', tau);
% 
% 
% %x0mod = x0.*(abs(x0) > options.vapnikEps);
% %xL1mod = xL1(1:n).*(abs(x0) > options.vapnikEps);
% %xVapnikmod = xVapnik(1:n).*(abs(x0)>options.vapnikEps);
% x0mod = x0;
% xVapnikmod = xVapnik;
% xL1mod = xL1;
% 
% figure(1)
% plot(1:n, x0mod, 1:n, xL1mod -2, 1:n, xVapnikmod + 2);legend('true', 'l1', 'vapnik')
% 
% 
% 
% figure(2)
% plot(1:m, b - A*x0, 1:m, b - A*xL1(1:n)+.05, 1:m, b - A*xVapnik(1:n)-.05);legend('True Outliers', 'l1 residuals', 'Vapnik Residuals')

%%
reset(RandStream.getDefaultStream);

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.


scale = 1e2;
s = exp(linspace(log(scale),log(1),m));
A = randn(m,m);  [Q,R] = qr(A,0);  U = Q;
A = randn(n,n);  [Q,R] = qr(A,0);  V = Q(:,1:m);  
A = U*diag(s)*V';

sigma = 0.05;

p  = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
w = 0.05 * sign(randn(n, 1));
x0 = x0 + w; % add some random noise
e = sigma * randn(m,1);
b  = A*x0 + e;

options.primal = 'lsq';
options.rootFinder = 'newton';
options.exact = 1;

options.vapnikEps = 0;

sigmaModel = 0.5*m*(sigma)^2;

xL1 = gbpdn(A, b, [], sigmaModel, [], options); % Find BP sol'n.

options.vapnikEps = 0.05;
xVapnik = gbpdn(A, b, [], sigmaModel, [], options); % Find BP sol'n.

%x0mod = x0.*(abs(x0) > options.vapnikEps);
%xL1mod = xL1(1:n).*(abs(x0) > options.vapnikEps);
%xVapnikmod = xVapnik(1:n).*(abs(x0)>options.vapnikEps);
x0mod = x0;
xVapnikmod = xVapnik;
xL1mod = xL1;

figure(1)
plot(1:n, x0mod, 1:n, xL1mod -2, 1:n, xVapnikmod + 2);legend('true', 'l1', 'vapnik')



figure(2)
plot(1:m, b - A*x0, 1:m, b - A*xL1(1:n)+.05, 1:m, b - A*xVapnik(1:n)-.05);legend('True Outliers', 'l1 residuals', 'Vapnik Residuals')
