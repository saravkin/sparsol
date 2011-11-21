function ok = oneProjectorHuberTest()

ok = 1;
problemSize = 1000;
x = randn(problemSize, 1); % generate random projector

eps = 0.001;    % small enough to matter

tau  = 100;    % 1-norm of x is over 500

y = oneProjectorHuber(x, 1, tau, eps);
hNorm = hubers(y, eps);

ok = ok && abs(hNorm-tau) <= 1e-3;

% NOTE: hubers(y, eps) = huber(y, eps)/(2*eps)


% using cvx to REALLY do the projection

%huberFunc = @(x)hubers(x, eps, m);

expID = 3;
cvx_begin
variable xs(problemSize)
minimize( norm( xs - x, 2 ) )
subject to
sum(huber(xs, eps)/(2*eps)) <= tau; % see NOTE
cvx_end

plot(1:1000, y - xs)

ok = ok && norm(y - xs, inf) <= 1e-4;


end
% 
% function vNorm = vNormCalc(y, vapnikEps)
% yMod = abs(y) - vapnikEps;
% yMod = max(yMod, 0);
% vNorm = sum(yMod);
% end