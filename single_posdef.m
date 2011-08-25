function [s] = single_posdef(n)
     x=rand(n);
     p=single(eye(n)*0.001+ x*x');
     %s = triu(p) + triu(p)';
     chol(p);
end
