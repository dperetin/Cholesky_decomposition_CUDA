#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>

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

void printMatrix(float * m, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			cout << m[i * size + j] << " ";
		cout << endl;
	}
}

void cpu_dpotrf_old(float *m_in, float *m_out, int size)
{
	for (int i = 0; i < size; i++) {
		float sum = 0;
		for (int k = 0; k < i; k++){
			sum += (m_out[k * size + i] * m_out[k * size + i]);
		}
		m_out[i * size + i] = sqrt(m_in[i * size + i] - sum);
		for (int j = i + 1; j < size; j++ ) {
			sum = 0;
			for (int k = 0; k < i; k++){
				sum += (m_out[k * size + i] * m_out[k * size + j]);
			}
			m_out[i * size + j] = (m_in[i * size + j] - sum) / m_out[i * size + i];
		}
	}
}

void cpu_dpotrf(float *m_in, float *m_out, int size, int p)
{
	for (int i = 0; i < 16; i++) {
			float sum = 0;
			for (int k = 0; k < i; k++){
				sum += (m_in[(k + p * 16) * size + (i + p * 16)] 
						* m_in[(k + p * 16) * size + (i + p * 16)]);
			}
			m_in[(i + p * 16) * size + (i + p * 16)] = 
				sqrt(m_in[(i + p * 16) * size + (i + p * 16)] - sum);
			for (int j = i + 1; j < 16; j++ ) {
				sum = 0;
				for (int k = 0; k < i; k++){
					sum += (m_in[(k + p * 16) * size + (i + p * 16)] 
						* m_in[(k + p * 16) * size + (j + p * 16)]);
				}
				m_in[(i + p * 16) * size + (j + p * 16)] = 
					(m_in[(i + p * 16) * size + (j + p * 16)] - sum) / 
							m_in[(i + p * 16) * size + (i + p * 16)];
			}
		}
}


__global__ void gpu_dpotrf(float *m_in, float *m_out, int size, int p)
{
	if(threadIdx.x == 0){
		for (int i = 0; i < 16; i++) {
			float sum = 0;
			for (int k = 0; k < i; k++){
				sum += (m_out[(k + p * 16) * size + (i + p * 16)] 
						* m_out[(k + p * 16) * size + (i + p * 16)]);
			}
			m_out[(i + p * 16) * size + (i + p * 16)] = 
				sqrt(m_in[(i + p * 16) * size + (i + p * 16)] - sum);
			for (int j = i + 1; j < 16; j++ ) {
				sum = 0;
				for (int k = 0; k < i; k++){
					sum += (m_out[(k + p * 16) * size + (i + p * 16)] 
						* m_out[(k + p * 16) * size + (j + p * 16)]);
				}
				m_out[(i + p * 16) * size + (j + p * 16)] = 
					(m_in[(i + p * 16) * size + (j + p * 16)] - sum) / 
							m_out[(i + p * 16) * size + (i + p * 16)];
			}
		}
	}
}

__global__ void gpu_inv(float *u, float *b, int size, int p)
{
	int tid = threadIdx.x, i, j;
	b[15 * 16 + tid] = b[15 * 16 + tid] / 
		u[(15 + p * 16) * size + (15 + 16 * p)];
	for (i = 14; i >= 0; i--) {
		for (j = i + 1; j < 16; j++) {
			b[i * 16 + tid] = b[i * 16 + tid] - 
							  u[(i + p * 16) * size + (j + p * 16)] * 
							  b[j * 16 + tid];
		}
		b[i * 16 + tid] = b[i * 16 + tid] / u[(i + p * 16) * size + 
							(i + 16 *p)];
	}
}

__global__ void gpu_inv_l(float *u, float *b, int size, int p)
{
	int i, j;
	int tid = threadIdx.x;
	b[0 * 16 + tid] = b[0 * 16 + tid] / 
		u[(0 + p * 16) * size + (0 + 16 * p)];
	for (i = 1; i < 16; i++){
		for (j = 0; j < i; j++){
			b[i * 16 + tid] = b[i * 16 + tid] - 
				u[(j + p * 16) * size + (i + p * 16)] *
				b[j * 16 + tid];
		}
		b[i * 16 + tid] = b[i * 16 + tid] / u[(i + p * 16) * size + 
							(i + 16 *p)];

	}
}

void cpu_inv(float *u, float *b)
{
	int tid, i, j;
	for (tid = 0; tid < 16; tid++){
	b[tid * 16 + 15] = b[tid * 16 + 15] / u[15 * 16 + 15];
	for (i = 14; i >= 0; i--) {
		for (j = i + 1; j < 16; j++) {
			b[tid * 16 + i] = b[tid * 16 + i] - 
							  u[i * 16 + j] * b[tid * 16 + j];
		}
		b[tid * 16 + i] = b[tid * 16 + i] / u[i * 16 + i];
	}
	}
}

__global__ void gpu_mm(float *m, float *a, float *b)
{
	__shared__ float s_a[16][16];
	__shared__ float s_b[16][16];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i;

	s_a[ty][tx] = a[ty * 16 + tx];
	s_b[ty][tx] = b[ty * 16 + tx];

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		m[ty * 16 + tx] = m[ty * 16 + tx] - s_a[i][ty] * s_b[i][tx];
	}
}

__global__ void gpu_mm_a(float *m, float *a, int p)
{
	__shared__ float s_a[16][16];
//	__shared__ float s_b[16][16];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i;

	s_a[ty][tx] = a[(ty) * 32 + tx + 16];
//	s_b[ty][tx] = a[(ty + 16) * 16 + tx + 16];

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		m[(ty + 16) * 32 + tx + 16] -= s_a[i][ty] * s_a[i][tx];
	}
}

/*
   Mnozi redak 16x16 matrica
 */
__global__ void gpu_mm_r(float *m, float *a, float *b)
{
	__shared__ float s_a[16][16];
	__shared__ float s_b[16][16];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i;

	s_a[ty][tx] = a[ty * 16 + tx];
	s_b[ty][tx] = b[ty * 32 + tx + 16];

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		m[ty * 32 + tx + 16] += s_a[ty][i] * s_b[i][tx];
	}
}


__global__ void gpu_dscal(float *v, int n)
{
	unsigned int i = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if(i < n) {
		v[i + 1] /= v[0];
	}
}

__global__ void gpu_sqrt(float *device_m)
{
	if(threadIdx.x == 0)
		device_m[0] = sqrt(device_m[0]);
}

__global__ void gpu_daxpy(float *a, float *b, int n, float *x, int k)
{
	//unsigned int tid = threadIdx.x;
	unsigned int i = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if(i < n) {
		a[i] -= (b[i] * x[k - 1]);
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
	int size = 32;//atoi(argv[2]);
	unsigned int timer, timer2, t=0, t2=0;
//	float *m, *a, *b;
//	float *d_m, *d_a, *d_b;
	float *m_in, *m_out, *device_m, *device_m_out, *eye, *device_eye;
	m_in = new float[size * size];
	m_out = new float[size * size];
	eye = new float[16 * 16];

	memset(m_out, 0, size * size * sizeof(float));
	memset(eye, 0, 16 * 16 * sizeof(float));

	init_eye(eye, 16);

	
/*	m = new float[size * size];
	a = new float[size * size];
	b = new float[size * size];*/

	int deviceOrdinal = 0;
	cudaSetDevice(deviceOrdinal);
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, deviceOrdinal);
	printf("%s\n\n", device_properties.name);

	printf("Ucitavanje iz matrice: ");

	CUT_SAFE_CALL(cutCreateTimer(&t));
	CUT_SAFE_CALL(cutStartTimer(t));
	
	loadMatrix(m_in, "po32.mat", size);
	/*loadMatrix(m, "m.mat", size);
	loadMatrix(a, "a.mat", size);
	loadMatrix(b, "b.mat", size);*/
	CUT_SAFE_CALL(cutStopTimer(t));

	printf("%f\n", cutGetTimerValue(t));

	


	timer = 0;
	timer2 = 0;

	printf("CPU racuna: ");

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	//cpu_dpotrf(m_in, m_out, size);

	CUT_SAFE_CALL(cutStopTimer(timer));

	printf("%f\n", cutGetTimerValue(timer));

	
	// GPU //
	int n = size;
	dim3 blokovaPoGridu, thredovaPoBloku;
	
	thredovaPoBloku.x = 16;
	thredovaPoBloku.y = 16;
	/*int x = n / 256;
	int z = (int)sqrt(x + 0.0) + 1;
	blokovaPoGridu.x = z;
	blokovaPoGridu.y = z;*/
	

	cudaMalloc((void **) &device_m, n * n * sizeof(float));
	cudaMalloc((void **) &device_m_out, n * n * sizeof(float));
	cudaMalloc((void **) &device_eye, 16 * 16 * sizeof(float));

	cudaMemset(device_m_out, 0, n * n *sizeof(float));
	
	/*cudaMalloc((void **) &d_m, n * n * sizeof(float));
	cudaMalloc((void **) &d_a, n * n * sizeof(float));
	cudaMalloc((void **) &d_b, n * n * sizeof(float));*/
	printf("Kopiranje matrice na GPU: ");

	CUT_SAFE_CALL(cutCreateTimer(&t2));
	CUT_SAFE_CALL(cutStartTimer(t2));

	cudaMemcpy(device_m, m_in, n * n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(device_eye, eye, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);
/*	cudaMemcpy(d_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n * n * sizeof(float), cudaMemcpyHostToDevice);
*/
	CUT_SAFE_CALL(cutStopTimer(t2));

	printf("%f\n", cutGetTimerValue(t2));

	//m_in[0] = sqrt(m_in[0]);

	printf("GPU racuna: \n");



	gpu_dpotrf<<<1, 1>>>(device_m, device_m_out, size, 0);
	cudaThreadSynchronize();
	gpu_inv_l<<<1, 16>>>(device_m_out, device_eye, size, 0);
	gpu_mm_r<<<1, thredovaPoBloku, 2 * 16 * 16 * sizeof(float)>>>
		(device_m_out, device_eye, device_m);
	gpu_mm_a<<<1, thredovaPoBloku, 2 * 16 * 16 * sizeof(float)>>>
		(device_m, device_m_out, 1);
	gpu_dpotrf<<<1, 1>>>(device_m, device_m_out, size, 1);
	//cpu_dpotrf(m_in, m_out, size, 0);

	//cudaThreadSynchronize();
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutCreateTimer(&timer2));
	CUT_SAFE_CALL(cutStartTimer(timer2));
	//gpu_inv<<<1, 16>>>(device_m_out, device_eye);
	//cpu_inv(m_out, eye);
	/*for (int i = 0; i < 1000000; i++)
	gpu_mm<<<1, thredovaPoBloku, 2 * n * n * sizeof(float)>>>(d_m, d_a, d_b);
	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(timer2));

	printf("%f\n", cutGetTimerValue(timer2));*/

//	cudaMemcpy(eye, device_eye, 
//			16 * 16 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_out, device_m_out, 
			n * n * sizeof(float), cudaMemcpyDeviceToHost);
	

	saveMatrix(m_out, "rez.mat", n);

	free(m_in);
	free(m_out);
	free(eye);
	cudaFree(device_m);
	cudaFree(device_m_out);
	cudaFree(device_eye);
/*	free(a);
	free(b);
	free(m);
	cudaFree(d_m);
	cudaFree(d_a);
	cudaFree(d_b);*/

	return 0;
}
