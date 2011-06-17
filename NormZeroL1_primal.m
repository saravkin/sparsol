function p = NormZeroL1_primal(x,weights)

p = norm(x.*weights,1);
