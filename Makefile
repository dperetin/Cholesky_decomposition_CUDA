test : main.cu 
	nvcc main.cu -lhdf5 -arch=sm_20 -o GPUCHOL -O3
