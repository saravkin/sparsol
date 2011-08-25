function ok = oneProjectorVapTest()

ok = 1;
n = 1000;
x = randn(n, 1); % generate random projector

vapnikEps = 0.01;    % small enough to matter

tau  = 100;    % 1-norm of x is over 500

y = oneProjectorVap(x, 1, tau, vapnikEps);
vNorm = vNormCalc(y, vapnikEps);

ok = ok && abs(vNorm-tau) <= 1e-8;

ySub = oneProjectorVap(x, 1, tau-n*vapnikEps, vapnikEps);
yOne = oneProjectorVap(x, 1, tau, 0);

plot(1:n, yOne, 1:n, ySub + 3);
legend('One proj', 'Vapnik Proj');


end

function vNorm = vNormCalc(y, vapnikEps)
yMod = abs(y) - vapnikEps;
yMod = max(yMod, 0);
vNorm = sum(yMod);
end