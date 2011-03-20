function [p] = posdef(n)
	x=rand(n);
	p=eye(n)*0.001+ x*x';
end
