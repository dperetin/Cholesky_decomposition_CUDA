function [] = bench()

k = 1;
for i=4:21
    size = 2^i;
    if i > 10
        size = 1024 * (i-9);
    end
    size
    filename = strcat(num2str(size), 's.h5');
    filename_rez = strcat('rez', num2str(size), 's.h5');
    x = hdf5read(filename', '/16');
    rez_gpu = hdf5read(filename_rez, '/16');
    
    
    time = 0;
    for j = 1:10
        tic;
        c = chol(x);
        time = time + toc;
    end
    b(k, 1) = size;
    b(k, 2) = (time/10)*1000;
    b(k, 3) = c(end, end);
    
    
    
    b(k, 4) = max(max(abs(x-c'*c)));
    
    gpu_c = tril(rez_gpu);
    b(k, 5) = max(max(abs(x-gpu_c*gpu_c')));
    k = k+1;
end
save '-ascii' 'single_rez.txt' b;
end

