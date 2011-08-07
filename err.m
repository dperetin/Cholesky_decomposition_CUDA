function err(m, r)
c=chol(m);
sum(sum(c))-sum(sum(r))
end
