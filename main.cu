#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>

#define N 128
#define BLOCK_SIZE 16

using namespace std;

void loadMatrix(float * matrix, char *s, int size)
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

void saveMatrix(float * matrix, char *s, int size)
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


__global__ void gpu_dpotrf(float *m_in, float *m_out, int size, int p)
{
	if(threadIdx.x == 0){
		for (int i = 0; i < BLOCK_SIZE; i++) {
			float sum = 0;
			for (int k = 0; k < i; k++){
				sum += (m_out[(k + p) * size + (i + p)] 
						* m_out[(k + p) * size + (i + p)]);
			}
			m_out[(i + p) * size + (i + p)] = 
				sqrt(m_in[(i + p) * size + (i + p)] - sum);
			for (int j = i + 1; j < BLOCK_SIZE; j++ ) {
				sum = 0;
				for (int k = 0; k < i; k++){
					sum += (m_out[(k + p) * size + (i + p)] 
						* m_out[(k + p) * size + (j + p)]);
				}
				m_out[(i + p) * size + (j + p)] = 
					(m_in[(i + p) * size + (j + p)] - sum) / 
							m_out[(i + p) * size + (i + p)];
			}
		}
	}
}

__global__ void gpu_inv_l(float *u, float *b, int size, int p)
{
	int i, j;
	int tid = threadIdx.x;
	b[tid] = b[+ tid] / 
		u[(p) * size + (p)];
	for (i = 1; i < BLOCK_SIZE; i++){
		for (j = 0; j < i; j++){
			b[i * BLOCK_SIZE + tid] = b[i * BLOCK_SIZE + tid] - 
				u[(j + p) * size + (i + p)] *
				b[j * BLOCK_SIZE + tid];
		}
		b[i * BLOCK_SIZE + tid] = b[i * BLOCK_SIZE + tid] / u[(i + p) * size + 
							(i + p)];

	}
}

__global__ void gpu_mm_a(float *m, float *a, int size, int p)
{
	__shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int i;

	s_a[ty][tx] = a[(ty + p) * size + tx + (p + BLOCK_SIZE) + bx * BLOCK_SIZE];
	s_b[ty][tx] = a[(ty + p) * size + tx + (p + BLOCK_SIZE) + by * BLOCK_SIZE];

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < BLOCK_SIZE; i++)
	{
		m[(ty + (p + BLOCK_SIZE + bx * BLOCK_SIZE)) * size + tx + (p + BLOCK_SIZE + by *BLOCK_SIZE)] -= s_a[i][ty] * s_b[i][tx];
	}
}

/*   
   Mnozi redak 16x16 matrica, m += a'b 
 */
__global__ void gpu_mm_r(float *m, float *a, float *b, int size, int p)
{
	__shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int stride = blockIdx.x + 1;
	int i;

	s_a[ty][tx] = a[ty * BLOCK_SIZE + tx];
	s_b[ty][tx] = b[(ty + p) * size + tx + (BLOCK_SIZE*stride + p)];

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < BLOCK_SIZE; i++)
	{
		m[(ty + p) * size + tx + (BLOCK_SIZE*stride + p)] += s_a[ty][i] * s_b[i][tx];
	}
}

void init_eye(float *v, int n)
{
	int i;
	for (i = 0; i < n; i++)
		v[i * n + i] = 1.;
}

int main(int argc, char *argv[])
{
	int size = 64;
	unsigned int timer2 = 0, t = 0, t2 = 0;

	float *m_in, *m_out, *device_m, *device_m_out, *eye, *device_eye;
	m_in = new float[size * size];
	m_out = new float[size * size];
	eye = new float[16 * 16];

	memset(m_out, 0, size * size * sizeof(float));
	memset(eye, 0, 16 * 16 * sizeof(float));

	init_eye(eye, 16);


	int deviceOrdinal = 0;
	cudaSetDevice(deviceOrdinal);
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, deviceOrdinal);
	printf("%s\n\n", device_properties.name);

	printf("Ucitavanje iz matrice: ");

	CUT_SAFE_CALL(cutCreateTimer(&t));
	CUT_SAFE_CALL(cutStartTimer(t));
	
	loadMatrix(m_in, "matrice/po64.mat", size);

	CUT_SAFE_CALL(cutStopTimer(t));

	printf("%f\n", cutGetTimerValue(t));

	// GPU //
	int n = size;
	dim3 blokovaPoGridu, thredovaPoBloku;
	
	thredovaPoBloku.x = 16;
	thredovaPoBloku.y = 16;
	blokovaPoGridu.x = 3;
	blokovaPoGridu.y = 3;


	cudaMalloc((void **) &device_m, n * n * sizeof(float));
	cudaMalloc((void **) &device_m_out, n * n * sizeof(float));
	cudaMalloc((void **) &device_eye, 16 * 16 * sizeof(float));

	cudaMemset(device_m_out, 0, n * n *sizeof(float));
	

	printf("Kopiranje matrice na GPU: ");

	CUT_SAFE_CALL(cutCreateTimer(&t2));
	CUT_SAFE_CALL(cutStartTimer(t2));

	cudaMemcpy(device_m, m_in, n * n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(device_eye, eye, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);

	CUT_SAFE_CALL(cutStopTimer(t2));

	printf("%f\n", cutGetTimerValue(t2));

	printf("GPU racuna: ");

	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutCreateTimer(&timer2));
	CUT_SAFE_CALL(cutStartTimer(timer2));

	int i;
	int it = n / 16 - 1;
	gpu_dpotrf<<<1, 1>>>(device_m, device_m_out, size, 0);

	for (i = 0; i < n / 16 - 1; i++) {
		cudaMemcpy(device_eye, eye, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);
		blokovaPoGridu.x = it;
		blokovaPoGridu.y = it;
		gpu_inv_l<<<1, 16>>>(device_m_out, device_eye, size, i * BLOCK_SIZE);
		gpu_mm_r<<<it, thredovaPoBloku, 2 * 16 * 16 * sizeof(float)>>>
			(device_m_out, device_eye, device_m, size, i * BLOCK_SIZE);
		gpu_mm_a<<<blokovaPoGridu, thredovaPoBloku, 2 * 16 * 16 * sizeof(float)>>>
		(device_m, device_m_out, size, i);
		gpu_dpotrf<<<1, 1>>>(device_m, device_m_out, size, (i + 1) * BLOCK_SIZE);
		it--;
		
	}

	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(timer2));

	printf("%f\n", cutGetTimerValue(timer2));

	
	cudaMemcpy(m_out, device_m_out, 
			n * n * sizeof(float), cudaMemcpyDeviceToHost);
	

	saveMatrix(m_out, "rez.mat", n);

	free(m_in);
	free(m_out);
	free(eye);
	cudaFree(device_m);
	cudaFree(device_m_out);
	cudaFree(device_eye);

	return 0;
}
