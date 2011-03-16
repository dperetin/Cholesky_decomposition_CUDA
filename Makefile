SDK_INC = $(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
SDK_LIB = $(HOME)/NVIDIA_GPU_Computing_SDK/C/lib

test : main.cu 
	nvcc main.cu -I$(SDK_INC) -L$(SDK_LIB) -lcutil -arch=sm_13 -o GPUCHOL -O3 
