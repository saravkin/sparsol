m = 1200; n = 5120; k = 200; % m rows, n cols, k nonzeros.
    p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
    A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
    b  = A*x0 + 0.005 * randn(m,1);
    
    options.solver = 1;
    
    options.rootFinder = 'newton';     
    [xNewton,info] = gbpdn(A, b, 0, 1e-1, [], options); % Find BP sol'n.
    
    options.rootFinder = 'secant';     
    [xSecant,info] = gbpdn(A, b, 0, 1e-1, [], options); % Find BP sol'n.