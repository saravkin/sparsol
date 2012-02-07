close all;
clear all;
m = 12; n = 51; k = 3; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
y  = A*x0 + 0.005 * randn(m,1);


taus = 0:0.2:4;
lambdas = 0:.01:0.6;

lenLam = length(lambdas);
lenTau = length(taus);

lambdaVals = zeros(lenLam, 1);
tauVals = zeros(lenTau, 1);

problemSize = n;

for i = 1:lenLam
    cvx_begin quiet
    variable xs(problemSize)
    minimize( 0.5*square_pos(norm(A*xs - y, 2)) + lambdas(i)*norm(xs, 1))
    cvx_end
    lambdaVals(i) = 0.5*norm(A*xs - y, 2) + lambdas(i)*norm(xs, 1);
end

figure()
plot(lambdas, lambdaVals)

for i = 1:lenTau
    cvx_begin quiet
    variable xs(problemSize)
    minimize( 0.5*square_pos(norm(A*xs - y, 2)))
    subject to
       norm(xs, 1) <= taus(i);
    cvx_end
    tauVals(i) = 0.5*norm(A*xs - y, 2);
end

figure()
plot(taus, tauVals)

