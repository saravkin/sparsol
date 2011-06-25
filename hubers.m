function [y g] = hubers(x, M, t)


%       HUBER(X) = 0.5*|X|^2   if |X|<=1,
%                  |X|-0.5  if |X|>=1.

% HUBER(X,M) is the Huber penalty function of halfwidth M, M.^2.*HUBER(X./M).
%   M must be real and positive.

%       HUBER(X,M,T) = T.*M.^2.*HUBER(X./(T*M)) if T > 0
%                      +Inf             if T <= 0

y = 0.5*sum(huber_s(abs(x), M, t));
%y = 0.5*sum(huber_s(abs(x)./t, M, t));

if(nargout > 1) % only compute gradient if necessary
    g = t.*M.*dhuber(x./(M));
    %g = M.*dhuber(x./(t.*M));
end
end

function [g] = dhuber(x)
g = x.*(abs(x) < 1) + (x >= 1) - (x <= -1);
end

function y = huber_s( x, M, t )

y = max( x, 0 );
z = min( y, M );
y = t .* z .* ( 2 * y - z );
q = t <= 0;
if nnz( q ),
    if length( t ) == 1,
        y = Inf * ones( sy );
    else
        y( q ) = Inf;
    end
end


end
