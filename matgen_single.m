function [] = matgen_single()

for i=4:21
    size = 2^i;
    if i > 10
        size = 1024 * (i-9);
    end
    size
    x = single_posdef(size);
    filename = strcat(num2str(size), 's.h5');
    hdf5write(filename, '/16', x)
    
end

end

