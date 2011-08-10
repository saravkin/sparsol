function ok = oneProjectorVapTest()

ok = 1;
problemSize = 1000;
x = randn(problemSize, 1); % generate random projector

vapnikEps = 0.05;    % small enough to matter

tau  = 100;    % 1-norm of x is over 500

y = oneProjectorVap(x, 1, tau, vapnikEps);
vNorm = vNormCalc(y, vapnikEps);

ok = ok && abs(vNorm-tau) <= 1e-8;



%% using cvx to REALLY do the projection

vNormFunc = @(x)vNormCalc(x, vapnikEps);

expID = 3;
cvx_begin
variable xs(problemSize)
minimize( norm( xs - x, 2 ) )
subject to
vNormFunc(xs) <= tau;
cvx_end

ok = ok && norm(y - xs, inf) <= 1e-4;


end

function vNorm = vNormCalc(y, vapnikEps)
yMod = abs(y) - vapnikEps;
yMod = max(yMod, 0);
vNorm = sum(yMod);
end