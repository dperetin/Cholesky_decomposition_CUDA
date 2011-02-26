#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>

using namespace std;

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

void cpu_dpotrf_old(double *m_in, double *m_out, int size)
{
	for (int i = 0; i < size; i++) {
		double sum = 0;
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

void cpu_dpotrf(double *m_in, double *m_out, int size, int p)
{
	for (int i = 0; i < 16; i++) {
			double sum = 0;
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


__global__ void gpu_dpotrf(double *m_in, double *m_out, int size, int p)
{
	if(threadIdx.x == 0){
		for (int i = 0; i < 16; i++) {
			double sum = 0;
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

__global__ void gpu_inv(double *u, double *b, int size, int p)
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

__global__ void gpu_inv_l(double *u, double *b, int size, int p)
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

void cpu_inv(double *u, double *b)
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

__global__ void gpu_mm(double *m, double *a, double *b)
{
	__shared__ double s_a[16][16];
	__shared__ double s_b[16][16];
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

__global__ void gpu_mm_a(double *m, double *a, int size, int p)
{
	__shared__ double s_a[16][16];
	__shared__ double s_b[16][16];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int i;

	s_a[ty][tx] = a[(ty + p * 16) * size + tx + (p + 1) * 16 + bx * 16];
	s_b[ty][tx] = a[(ty + p * 16) * size + tx + (p + 1) * 16 + by * 16];

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		m[(ty + (p + 1 + bx) * 16) * size + tx + (p + 1 + by) * 16] -= s_a[i][ty] * s_b[i][tx];
	}
}

/*   
   Mnozi redak 16x16 matrica, m += a'b 
 */
__global__ void gpu_mm_r(double *m, double *a, double *b, int size, int p)
{
	__shared__ double s_a[16][16];
	__shared__ double s_b[16][16];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int stride = blockIdx.x + 1;
	int i;

	s_a[ty][tx] = a[ty * 16 + tx];
	s_b[ty][tx] = b[(ty + p * 16) * size + tx + 16 * (stride + p)];

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		m[(ty + p * 16) * size + tx + 16 * (stride + p)] += s_a[ty][i] * s_b[i][tx];
	}
}


__global__ void gpu_dscal(double *v, int n)
{
	unsigned int i = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if(i < n) {
		v[i + 1] /= v[0];
	}
}

__global__ void gpu_sqrt(double *device_m)
{
	if(threadIdx.x == 0)
		device_m[0] = sqrt(device_m[0]);
}

__global__ void gpu_daxpy(double *a, double *b, int n, double *x, int k)
{
	//unsigned int tid = threadIdx.x;
	unsigned int i = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if(i < n) {
		a[i] -= (b[i] * x[k - 1]);
	}
}

void init_eye(double *v, int n)
{
	int i;
	for (i = 0; i < n; i++)
		v[i * n + i] = 1.;
}

int main(int argc, char *argv[])
{
	int size = 4096;
	unsigned int timer, timer2, t=0, t2=0;

	double *m_in, *m_out, *device_m, *device_m_out, *eye, *device_eye;
	m_in = new double[size * size];
	m_out = new double[size * size];
	eye = new double[16 * 16];

	memset(m_out, 0, size * size * sizeof(double));
	memset(eye, 0, 16 * 16 * sizeof(double));

	init_eye(eye, 16);


	int deviceOrdinal = 0;
	cudaSetDevice(deviceOrdinal);
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, deviceOrdinal);
	printf("%s\n\n", device_properties.name);

	printf("Ucitavanje iz matrice: ");

	CUT_SAFE_CALL(cutCreateTimer(&t));
	CUT_SAFE_CALL(cutStartTimer(t));
	
	loadMatrix(m_in, "po4096.mat", size);

	CUT_SAFE_CALL(cutStopTimer(t));

	printf("%f\n", cutGetTimerValue(t));

	


	timer = 0;
	timer2 = 0;

	printf("CPU racuna: ");

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	//cpu_dpotrf_old(m_in, m_out, size);

	CUT_SAFE_CALL(cutStopTimer(timer));

	printf("%f\n", cutGetTimerValue(timer));

	
	// GPU //
	int n = size;
	dim3 blokovaPoGridu, thredovaPoBloku;
	
	thredovaPoBloku.x = 16;
	thredovaPoBloku.y = 16;
	blokovaPoGridu.x = 3;
	blokovaPoGridu.y = 3;


	cudaMalloc((void **) &device_m, n * n * sizeof(double));
	cudaMalloc((void **) &device_m_out, n * n * sizeof(double));
	cudaMalloc((void **) &device_eye, 16 * 16 * sizeof(double));

	cudaMemset(device_m_out, 0, n * n *sizeof(double));
	

	printf("Kopiranje matrice na GPU: ");

	CUT_SAFE_CALL(cutCreateTimer(&t2));
	CUT_SAFE_CALL(cutStartTimer(t2));

	cudaMemcpy(device_m, m_in, n * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(device_eye, eye, 16 * 16 * sizeof(double), cudaMemcpyHostToDevice);

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
		cudaMemcpy(device_eye, eye, 16 * 16 * sizeof(double), cudaMemcpyHostToDevice);
		blokovaPoGridu.x = it;
		blokovaPoGridu.y = it;
		gpu_inv_l<<<1, 16>>>(device_m_out, device_eye, size, i);
		gpu_mm_r<<<it, thredovaPoBloku, 2 * 16 * 16 * sizeof(double)>>>
			(device_m_out, device_eye, device_m, size, i);
		gpu_mm_a<<<blokovaPoGridu, thredovaPoBloku, 2 * 16 * 16 * sizeof(double)>>>
		(device_m, device_m_out, size, i);
		gpu_dpotrf<<<1, 1>>>(device_m, device_m_out, size, i + 1);
		it--;
		
	}

	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(timer2));

	printf("%f\n", cutGetTimerValue(timer2));

	
	cudaMemcpy(m_out, device_m_out, 
			n * n * sizeof(double), cudaMemcpyDeviceToHost);
	

	saveMatrix(m_out, "rez.mat", n);

	free(m_in);
	free(m_out);
	free(eye);
	cudaFree(device_m);
	cudaFree(device_m_out);
	cudaFree(device_eye);

	return 0;
}
