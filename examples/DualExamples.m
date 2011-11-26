 m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
    p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
    A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
    b  = A*x0 + 0.005 * randn(m,1);
    opts = spgSetParms('optTol',1e-4);
    [x,r,g,info] = spgl1(A, b, 0, 1e-3, [], opts); % Find BP sol'n.
    
    