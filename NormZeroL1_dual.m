function d = NormZeroL1_dual(x,weights)

idxnz = find(weights > 0);
d = norm(x(idxnz)./weights(idxnz),inf);
