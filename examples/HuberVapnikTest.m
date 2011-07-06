%reset(RandStream.getDefaultStream);

m = 120; n = 512;  % m rows, n cols
dWeight = 1./(1:n).^(1.5);
dWeight = dWeight';

p = randperm(n); x0 = zeros(n,1); x0(p(1:n)) = sign(randn(n,1));
x0(p(1:n)) = x0.*dWeight;

A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.005 * randn(m,1);
tau = norm(x0,1);

s = 5; % number of outliers
Err = zeros(m, 1); % outliers
pErr = randperm(m);
Err(pErr(1:s)) = .5*randn(s, 1); % outliers are 200 times the size
b = b + Err;



%% Subproblem options
options.vapnikEps = 0;
options.lassoOpts.optTol = 0;
options.solver = 1;     %1 for spg,  2 for pqn
options.lassoOpts.verbosity = 0;
options.tolerance = 1e-7*norm(b);
options.primal = 'lsq';
sigma = 1e-2;

%% Exact Newton, vapnik parameter = 0
options.rootFinder = 'newton';
options.exact = 1;
[xL1L2,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);


options.primal = 'huber';
options.hapaM = .005;

options.exact = 1;
[xL1Huber,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);

%% Exact Newton, vapnik parameter = 
options.vapnikEps = 0.005;
[xVapnikHuber,info] = gbpdn(A, b, 0, sigma, [], options); % Find BP sol'n.
fprintf('Target tau = %15.7e\n', tau);



figure(1)
plot(1:n, xVapnikHuber + 1, 1:n, xL1L2 -1, 1:n, xL1Huber , 1:n, x0 + 2);legend('VapHuber', 'L1L2', 'L1Huber', 'true')



figure(2)
plot(1:m, b - A*xVapnikHuber(1:n)+1, 1:m, b - A*xL1L2 - 1,  1:m, b - A*xL1Huber(1:n), 1:m, b - A*x0 + 2 );legend('VapHuber Res', 'L1L2Res', 'L1HuberRes', 'True')


truePeak = abs(x0) > .001; % take lots of peaks
xL1L2Found = abs(xL1L2)  > options.vapnikEps;
xL1L2Sens = 100*sum(truePeak.*xL1L2Found)/sum(truePeak);
xL1L2Spec = 100*sum((1-truePeak).*(1-xL1L2Found))/sum(1-truePeak);


xL1HuberFound = abs(xL1Huber)  > options.vapnikEps;
xL1HuberSens = 100*sum(truePeak.*xL1HuberFound)/sum(truePeak);
xL1HuberSpec = 100*sum((1-truePeak).*(1-xL1HuberFound))/sum(1-truePeak);


xVapnikHuberFound = abs(xVapnikHuber) > options.vapnikEps;
xVapnikHuberSens = 100*sum(truePeak.*xVapnikHuberFound)/sum(truePeak);
xVapnikHuberSpec = 100*sum((1-truePeak).*(1-xVapnikHuberFound))/sum(1-truePeak);



fprintf(' %-22s %-22s %-22s %-22s %-22s %-22s \n\n','Constraint:','Objective:','Inf-Norm:','2-Norm:','Sensitivity:', 'Specificity:');
fprintf(' %-22s %-22s %-22.2e %-22.2e %-22.2e %-22.2e \n','L1','L2',norm(x0 - xL1L2, inf),norm(x0 - xL1L2, 2), xL1L2Sens, xL1L2Spec);
fprintf(' %-22s %-22s %-22.2e %-22.2e %-22.2e %-22.2e\n','Huber','L1',norm(x0 - xL1Huber, inf),norm(x0 - xL1Huber, 2), xL1HuberSens, xL1HuberSpec);
fprintf(' %-22s %-22s %-22.2e %-22.2e %-22.2e %-22.2e\n','Huber','Vapnik',norm(x0 - xVapnikHuber, inf),norm(x0 - xVapnikHuber, 2), xVapnikHuberSens, xVapnikHuberSpec);