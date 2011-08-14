function [] = bench()

size = 16;
j = 1024;
k = 1;
for i=4:20
    size = 2^i;
    if i > 10
        size = 1024 * (i-9);
    end
    size
    x = single_posdef(size);
    filename = strcat(num2str(size), 's.h5');
    hdf5write(filename,'/16',x)
    
    time = 0;
    for j = 1:10
        tic;
        c = chol(x);
        time = time + toc;
    end
    b(k, 1) = size;
    b(k, 2) = time/10;
    b(k, 3) = c(end, end);
    k = k+1;
end
save '-ascii' 'rez.txt' b;
end

