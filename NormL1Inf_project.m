function X = NormL1Inf_project(groups,C,w,tau)

C = reshape(C,round(length(C)/groups),groups);

if tau == 0
  X = zeros(size(C));
  X = X(:);
  return;
end

if nargin < 2 || isempty(w)
   w = ones(size(C,1),1);
elseif isscalar(w)
   w = ones(size(C,1),1) * w;  
end


%tic
%% Compute cvx solution
%cvx_begin
%   variables Z(size(C));
%   minimize norm(Z-C,'fro');
%   subject to
%      sum(norms(diag(w) * Z,Inf,2)) <= tau
%cvx_end
%t1 = toc;

X = C;
n = sum(diag(w) * max(abs(X),[],2)); % Current norm

% Return immediately if C is already feasible
if n <= tau, X = X(:); return; end

%tic
% Sort each row
[X,perm] = sort(abs(C),2,'descend');

idx = ones(size(C,1),1);
v   = zeros(size(C,1),1);
m   = X(:,1);                          % Maximum absolute value
b   = X(:,1) - X(:,2);                 % Bound, if only one
                                       % element, set bound to value
wOld = w;

while(n > tau)
   lambda = ((n - tau) - sum(w.*v./idx)) / sum((w.^2)./idx);
   
   % Should check idx etc
   t = (lambda * w + v) ./ idx;
   
   if any(t > b)
      % Compute which entry requires minimum lambda for violation
      lambdaViol = (idx.*b - v)./(w);
      [dummy,j] = min(lambdaViol);

      % Update bounds
      v(j) = v(j) - idx(j) * b(j);
      n    = n - w(j) * b(j);
      m(j) = m(j) - b(j);
      
      idx(j) = idx(j) + 1;
      if idx(j) > size(C,2)
         b(j)   = 1;
         w(j)   = 0;
         v(j)   = 0;
      elseif idx(j)  == size(C,2)
         b(j)   = X(j,idx(j));
      else
         b(j)   = X(j,idx(j)) - X(j,idx(j)+1);
      end
   else
      m = m - t;
      n = n - w' * t;
      break;
   end
end

% Set solution
X = zeros(size(C));
for i=1:size(C,1)
   X(i,:) = sign(C(i,:)) .* min(abs(C(i,:)),m(i));
end
%t2 = toc;

%
%
%[norm(C-Z,'fro')-norm(C-X,'fro'), ...
% sum(diag(wOld) * max(abs(X),[],2))-tau, ...
% sum(diag(wOld) * max(abs(Z),[],2))-tau]
%
%[t1 t2]

X = X(:);
