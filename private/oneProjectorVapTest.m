function ok = oneProjectorVapTest()

x = randn(1000, 1); % generate random projector

vapnikEps = 0.05;    % small enough to matter 

tau  = 100;    % 1-norm of x is over 500

y = oneProjectorVap(x, 1, tau, vapnikEps);
vNorm = vNormCalc(y, vapnikEps);

ok = abs(vNorm-tau) <= 1e-8;
end

function vNorm = vNormCalc(y, vapnikEps)
yMod = abs(y) - vapnikEps;
yMod = yMod.*(yMod > 0);
vNorm = sum(yMod);

end