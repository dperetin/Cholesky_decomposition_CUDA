function matgen
	x = posdef(16);
	hdf5write('fullspread.h5', 'matrice/16', x);
	c=chol(x);
	hdf5write('fullspread.h5', 'matrice/r16', c(end, end),  'WriteMode', 'append');
	for i = 5:10
   		dim = 2^i	
		x = posdef(dim);
		hdf5write('fullspread.h5', strcat('matrice/',int2str(dim)), x, 'WriteMode', 'append');
		c=chol(x);
		hdf5write('fullspread.h5', strcat('matrice/r',int2str(dim)), c(end, end), 'WriteMode', 'append');

	end
	for i = 2048:1024:9216
   		dim = i	
		x = posdef(dim);
		hdf5write('fullspread.h5', strcat('matrice/',int2str(dim)), x, 'WriteMode', 'append');
		c=chol(x);
		hdf5write('fullspread.h5', strcat('matrice/r',int2str(dim)), c(end, end), 'WriteMode', 'append');
	end 
