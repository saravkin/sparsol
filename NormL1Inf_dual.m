function p = NormL1Inf_dual(groups,x,weights)

x = reshape(x,round(length(x)/groups),groups);

p = sum(abs(x),2);
if ~isempty(weights)
  if isscalar(weights)
     p = weights * max(p);
  else
     p = max(weights .* p);
  end
else
  p = max(p);
end
