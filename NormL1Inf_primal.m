function p = NormL1Inf_primal(groups,x,weights)

x = reshape(x,round(length(x)/groups),groups);

p = max(abs(x),[],2);
if ~isempty(weights)
  if isscalar(weights)
     p = weights * sum(p);
  else
     p = weights' * p;
  end
else
  p = sum(p);
end
