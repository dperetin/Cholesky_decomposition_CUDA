function [] = matgen_double()

for i=4:18
    size = 2^i;
    if i > 10
        size = 1024 * (i-9);
    end
    size
    x = posdef(size);
    filename = strcat(num2str(size), 'd.h5');
    hdf5write(filename, '/16', x)
    
end

end

