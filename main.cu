#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>
#include <hdf5.h>

using namespace std;

void cpu_potrf(double *m_in, double *m_out, int size)
{
	for (int i = 0; i < size; i++) {
		double sum = 0;
		for (int k = 0; k < i; k++) {
			sum += (m_out[k * size + i] * m_out[k * size + i]);
		}
		m_out[i * size + i] = sqrt(m_in[i * size + i] - sum);
		for (int j = i + 1; j < size; j++ ) {
			sum = 0;
			for (int k = 0; k < i; k++) {
				sum += (m_out[k * size + i] * m_out[k * size + j]);
			}
			m_out[i * size + j] = (m_in[i * size + j] - sum) / 
								   m_out[i * size + i];
		}
	}
}

void standard (double *A, double  *B, double *C, int size)
{
	int i, j, k;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			for (k = 0; k < size; k++) {
	            C[i * size + j] += A[k * size + i] * B[k * size + j]; 
				if (i==j)
					C[i*size+j]+= 0.001;
			}
}

void init(double *v, int n)
{
	int i;
	srand(time(NULL));
	for (i = 0; i < n; i++)
		v[i] = rand() / (double(RAND_MAX) + 1) - 1; 
}

void loadMatrix(double * matrix, char *s, int size)
{
	fstream f;
	int i = 0;
	f.open(s, ifstream::in);
	while (f.good()) {
		f >> matrix[i];
		i++;
	}
	f.close();
}

void saveMatrix(double * matrix, char *s, int size)
{
	fstream f;
	f.open(s, ifstream::out);
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			f << matrix[i * size + j] << " ";
		}
		f << endl;
	}
	f.close();
}

__global__ void gpu_potrf(double *m, int size, int p)
{
	int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double a[16][16 + 1];
    a[ty][tx] = m[(ty + 16 * p) * size + tx + 16 * p];

    __syncthreads();

    double d;

	#pragma unroll 16
    for (int k = 0; k < 16; k++) {
		__syncthreads();

		d = rsqrt(a[k][k]);

		__syncthreads();

		if ((ty == k) && (tx >= k)) 
	    	a[tx][ty] = (a[tx][ty]) * d;
	
		__syncthreads();

		if ((ty >= tx) && (tx > k)) 
	    	a[ty][tx] = a[ty][tx] - a[tx][k] * a[ty][k]; 
	}

    __syncthreads();

    if (ty >= tx) 
		m[(tx + 16 * p) * size + ty + 16 * p] = a[ty][tx];
    
}

__global__ void gpu_inv_l(double *u, int size, int p)
{
	int i, j;

	int tid = threadIdx.x;
	int bx = blockIdx.x + 1;

	__shared__ double b[16][16];

	for(i = 0; i < 16; i++)
		b[i][tid] = u[(i + p * 16) * size + tid + (bx + p) * 16];

	b[0][tid] = b[0][tid] /	u[(0 + p * 16) * size + (0 + 16 * p)];

	for (i = 1; i < 16; i++){
		for (j = 0; j < i; j++) {
			b[i][tid] = b[i][tid] - 
				u[(j + p * 16) * size + (i + p * 16)] * b[j][tid];
		}
		b[i][tid] = b[i][tid] / u[(i + p * 16) * size + (i + 16 *p)];
	}

	for(i = 0; i < 16; i++)
		u[(i + p * 16) * size + tid + (bx + p) * 16] = b[i][tid];
}

__global__ void gpu_mm_a(double *m, int size, int p, int it, int o, int e)
{
	__shared__ double s_a[16][16];
	__shared__ double s_b[16][16];
	//__shared__ double s_c[16][16];
	double s_c = 0;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx_g = blockIdx.x;
	int by_g = blockIdx.y;
	int i;

	int bx=0, by=0;

		int pi;

		pi = it - 1 - by_g;
		if(bx_g <= pi){
			by=by_g;
			bx=by_g+bx_g;
		}
		else{
			by=it-(by_g+o);
			bx=(it-by_g)+bx_g-pi+e;
		}

	s_a[ty][tx] = m[(ty + p * 16) * size + tx + (p + 1 + by) * 16];
	s_b[ty][tx] = m[(ty + p * 16) * size + tx + (p + 1 + bx) * 16];
	//s_c[ty][tx] = 0;

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		s_c += s_a[i][ty] * s_b[i][tx];
	}

	m[(ty + (p + 1 + by) * 16) * size + tx + (p + 1 + bx) * 16] -= s_c;
}

int main(int argc, char *argv[])
{

	if (argc < 2 || argc > 4)
	{
		fprintf(stderr, "GPUCHOL [red matrice] [opcionalno - ime datoteke]");
		return 1;
	}

	int size = atoi(argv[1]);

	unsigned int t_gpu = 0, t_mat_gen = 0, t_h2d = 0, 
				 t_cpu = 0, t_mat_load = 0, t_d2h;

	hid_t       file_id, dataset_id;
    
	double *m_in, *device_m, *v, *cpu_rez;
	m_in = new double[size * size];
//	m_out = new double[size * size];
	
	memset(m_in, 0, size * size * sizeof(double));
//	memset(m_out, 0, size * size * sizeof(double));
	
	

	int deviceOrdinal = 0;
	cudaSetDevice(deviceOrdinal);
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, deviceOrdinal);
	printf("\n%s\n\n", device_properties.name);


	if (argc == 2) {

		cpu_rez = new double[size * size];
		v = new double[size * size];
		memset(cpu_rez, 0, size * size * sizeof(double));

		printf("Generiranje matrice:\t\t");
		fflush(stdout);

		CUT_SAFE_CALL(cutCreateTimer(&t_mat_gen));
		CUT_SAFE_CALL(cutStartTimer(t_mat_gen));
	
		
		init(v, size * size);
		standard(v, v, m_in, size);

		CUT_SAFE_CALL(cutStopTimer(t_mat_gen));

		printf("%f\n", cutGetTimerValue(t_mat_gen));

		printf("CPU racuna:\t\t\t");

		CUT_SAFE_CALL(cutCreateTimer(&t_cpu));
		CUT_SAFE_CALL(cutStartTimer(t_cpu));
	
		cpu_potrf(m_in, cpu_rez, size);
		CUT_SAFE_CALL(cutStopTimer(t_cpu));

		printf("%f\n\n", cutGetTimerValue(t_cpu));

//		saveMatrix(m_in, "m.mat", size);
//		saveMatrix(cpu_rez, "cpu_rez.mat", size);
	}

	if (argc == 3) {

		printf("Ucitavanje matrice iz datoteke:\t");
		fflush(stdout);

		CUT_SAFE_CALL(cutCreateTimer(&t_mat_load));
		CUT_SAFE_CALL(cutStartTimer(t_mat_load));
	
		file_id = H5Fopen(argv[2], H5F_ACC_RDWR, H5P_DEFAULT);
    	dataset_id = H5Dopen(file_id, "/16", H5P_DEFAULT);
    	H5Dread(dataset_id, H5T_IEEE_F64LE, 
    					 H5S_ALL, H5S_ALL, H5P_DEFAULT, m_in);
        H5Dclose(dataset_id);
   		H5Fclose(file_id);
		
		CUT_SAFE_CALL(cutStopTimer(t_mat_load));

		printf("%f\n\n", cutGetTimerValue(t_mat_load));

	}

	// GPU //
	int n = size;
	
	dim3 blokovaPoGridu, thredovaPoBloku;
	
	thredovaPoBloku.x = 16;
	thredovaPoBloku.y = 16;

	cudaMalloc((void **) &device_m, n * n * sizeof(double));

	printf("Kopiranje matrice na GPU:\t");

	CUT_SAFE_CALL(cutCreateTimer(&t_h2d));
	CUT_SAFE_CALL(cutStartTimer(t_h2d));

	cudaMemcpy(device_m, m_in, n * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(t_h2d));

	printf("%f\n", cutGetTimerValue(t_h2d));

	printf("GPU racuna:\t\t\t");

	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutCreateTimer(&t_gpu));
	CUT_SAFE_CALL(cutStartTimer(t_gpu));

	int i, o, e;
	int it = n / 16 - 1;
	gpu_potrf <<<1, thredovaPoBloku>>> (device_m, size, 0);

	for (i = 0; i < n / 16 - 1; i++) {

		gpu_inv_l <<<it, 16>>> (device_m, size, i);
		
		if(it % 2){
			blokovaPoGridu.y = (it + 1) / 2;
			blokovaPoGridu.x = it;
			o = 0;
			e = -1;
		}
		else{
			blokovaPoGridu.y = it / 2;
			blokovaPoGridu.x = it + 1;
			o = 1;
			e = -2;
		}
		
		gpu_mm_a <<<blokovaPoGridu, thredovaPoBloku>>> (device_m, size, i, it, o, e);
	
		gpu_potrf <<<1, thredovaPoBloku>>> (device_m, size, i + 1);
	
		it--;
	}
	
	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(t_gpu));

	printf("%f\n", cutGetTimerValue(t_gpu));
	

	printf("Kopiranje matrice natrag:\t");
	CUT_SAFE_CALL(cutCreateTimer(&t_d2h));
	CUT_SAFE_CALL(cutStartTimer(t_d2h));

	cudaMemcpy(m_in, device_m, 
			n * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(t_d2h));

	printf("%f\n", cutGetTimerValue(t_d2h));
	printf("----------------------------------------------\n");
	printf("UKUPNO:\t\t\t\t%f\n\n", cutGetTimerValue(t_d2h) + 
						   cutGetTimerValue(t_h2d) + 
						   cutGetTimerValue(t_gpu));
	if (argc == 2) {
		printf("CPU %.13f\n", cpu_rez[(size - 1) * size + size - 1]);
	}
	printf("GPU %.13f\n", m_in[(size - 1) * size + size - 1]);


//	saveMatrix(m_out, "rez.mat", n);

	free(m_in);
//	free(m_out);
	if (argc == 2) {
		free(cpu_rez);
		free(v);
	}
	cudaFree(device_m);

	return 0;
}

