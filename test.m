function [c]=test(x)
	tic
	c=chol(x);
	toc
end

