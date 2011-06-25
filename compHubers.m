function [f r g y] = compHubers(x, Aprod, b, hparaM, hparaT)



M = hparaM;
T = hparaT;


%       HUBER(X) = 0.5*|X|^2   if |X|<=1,
%                  |X|-0.5  if |X|>=1.

% HUBER(X,M) is the Huber penalty function of halfwidth M, M.^2.*HUBER(X./M).
%   M must be real and positive.

%       HUBER(X,M,T) = T.*M.^2.*HUBER(X./(T*M)) if T > 0
%                      +Inf             if T <= 0


r = b - Aprod(x,1);

switch(nargout)
    case {2}
        f = hubers(r, M, T);
    case {3}
        [f gTemp] = hubers(r, M, T);
        g = -Aprod(gTemp, 2);
    case {4}
        [f gTemp] = hubers(r, M, T);
        g = -Aprod(gTemp, 2);
        %y         =   r.*(abs(r) <= (M*T)) + T*M^2*sign(r).*(abs(r) > (M*T));
        y = r.*(abs(r) <= (M)) + M*sign(r).*(abs(r) > (M));
end



end
