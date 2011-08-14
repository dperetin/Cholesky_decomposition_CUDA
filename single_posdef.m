function [s] = single_posdef(n)
     x=single(rand(n));
     p=single(eye(n)*0.001+ x*x');
     s = triu(p) + triu(p)';
end
