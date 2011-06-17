randn('state',0); rand('state',1);

% Set parameters
m = 128;
n = 128;
r =   7;
p = 0.3; % Fraction of entries to keep

% Create random rank r matrix B of size m by n
B  = randn(m,r) * randn(r,n);

% Select a percentage of entries
s = round(p*m*n);
idx = randperm(m*n); idx = idx(1:s);
mask = ones(m,n)*NaN; mask(idx) = 1;

% Create restriction matrix A
A = sparse(s,m*n); A((1:s) + (idx-1)*s) = 1;

% Call gbpdn
options.verbosity = 1;
[X,info] = gbpdn_nucnorm(A,A*B(:),m,n,0,options);

% Plot
figure(1);
subplot(2,2,1); pcolor(B),shading flat
                title('Original');
subplot(2,2,2); pcolor(B.*mask),shading flat
                title('Observed');
subplot(2,2,3); pcolor(X),shading flat
                title('Recovered');
subplot(2,2,4); pcolor(B-X),shading flat
                title('Difference');


