#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cutil.h>
#include <math.h>

using namespace std;

void cpu_dpotrf(float *m_in, float *m_out, int size)
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



void standard (float *A, float  *B, float *C, int size)
{
	    int i, j, k;

		    for (i = 0; i < size; i++)
				        for (j = 0; j < size; j++)
							            for (k = 0; k < size; k++)

										{

											                C[i * size + j] += A[k * size + i] * B[k * size + j]; 
															if (i==j)C[i*size+j]+= 0.001;
										}
}

void init(float *v, int n)
{
	    int i;
		    srand(time(NULL));
			    for (i = 0; i < n; i++)
					        v[i] = rand() / (float(RAND_MAX) + 1)-1; 
}



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


__global__ void gpu_dpotrf(float *m, int size, int p)
{
	int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float a[16][16+1];
    a[ty][tx] = m[(ty + 16 * p) * size + tx + 16 * p];

    __syncthreads();

    float fac;

	#pragma unroll 16
    for (int k = 0; k < 16; k++)
    {
		__syncthreads();
		fac = rsqrt(a[k][k]);
		__syncthreads();
		if ((ty == k) && (tx >= k)) 
	    	a[tx][ty] = (a[tx][ty]) * fac;
	
		__syncthreads();

		if ((ty >= tx) && (tx > k)) 
	    	a[ty][tx]=a[ty][tx] - a[tx][k]*a[ty][k]; 
	

    }

    __syncthreads();

    if (ty>=tx) 
	m[(tx+16*p)*size+ty+16*p]=a[ty][tx];
    


}

__global__ void gpu_inv_l(float *u, int size, int p)
{
	int i, j;

	int tid = threadIdx.x;
	int bx=blockIdx.x+1;
	__shared__ float b[16][16];
	for(i=0;i<16;i++)
		b[i][tid]=u[(i+p*16)*size+tid+(bx+p)*16];
	b[0][tid] = b[0][tid] / 
		u[(0 + p * 16) * size + (0 + 16 * p)];
	for (i = 1; i < 16; i++){
		for (j = 0; j < i; j++){
			b[i][tid] = b[i][tid] - 
				u[(j + p * 16) * size + (i + p * 16)] *
				b[j][tid];
		}
		b[i][tid] = b[i][tid] / u[(i + p * 16) * size + 
							(i + 16 *p)];

	}
	for(i=0;i<16;i++)
		u[(i+p*16)*size+tid+(bx+p)*16]=b[i][tid];
}

__global__ void gpu_mm_a(float *m, int size, int p, int it)
{
	__shared__ float s_a[16][16];
	__shared__ float s_b[16][16];
	__shared__ float s_c[16][16];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx_g = blockIdx.x;
	int by_g = blockIdx.y;
	int i;

	int bx=0, by=0, o = 0, e = -1;

		if(!(it % 2)){
			o = 1;
			e = -2;
		}

		int pi, n = it;

		pi = n - 1 - by_g;
		if(bx_g <= pi){
			by=by_g;
			bx=by_g+bx_g;
		}
		else{
			by=n-(by_g+o);
			bx=(n-by_g)+bx_g-pi+e;
		}

	s_a[ty][tx] = m[(ty + p * 16) * size + tx + (p + 1) * 16 + by * 16];
	s_b[ty][tx] = m[(ty + p * 16) * size + tx + (p + 1) * 16 + bx * 16];
	s_c[ty][tx] = 0;

	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		s_c[ty][tx] += s_a[i][ty] * s_b[i][tx];
	}

	m[(ty + (p + 1 + by) * 16) * size + tx + (p + 1 + bx) * 16] -= s_c[ty][tx];
}


__global__ void gpu_mm_r(float *a, float *b, int size, int p)
{
	__shared__ float s_a[16][16];
	__shared__ float s_b[16][16];
	__shared__ float s_c[16][16];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int stride = blockIdx.x + 1;
	int i;

	s_a[ty][tx] = a[ty * 16 + tx];
	s_b[ty][tx] = b[(ty + p * 16) * size + tx + 16 * (stride + p)];
	s_c[ty][tx] = 0;
	__syncthreads();

	#pragma unroll 16
	for (i = 0; i < 16; i++)
	{
		s_c[ty][tx] += s_a[ty][i] * s_b[i][tx];
	}
	b[(ty + p * 16) * size + tx + 16 * (stride + p)] = s_c[ty][tx];
}

void init_eye(float *v, int n)
{
	int i;
	for (i = 0; i < n; i++)
		v[i * n + i] = 1.;
}

int main(int argc, char *argv[])
{
	int size = 1024;

	unsigned int timer2 = 0, t = 0, t2 = 0, ta = 0;

	float *m_in, *m_out, *device_m, *device_m_out, *eye, *device_eye, *v, *cpu_rez;
	m_in = new float[size * size];
	m_out = new float[size * size];
	cpu_rez = new float[size * size];
	v = new float[size * size];
	eye = new float[16 * 16];

	memset(m_in, 0, size * size * sizeof(float));
	memset(m_out, 0, size * size * sizeof(float));
	memset(cpu_rez, 0, size * size * sizeof(float));
	memset(eye, 0, 16 * 16 * sizeof(float));

	init_eye(eye, 16);


	int deviceOrdinal = 0;
	cudaSetDevice(deviceOrdinal);
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, deviceOrdinal);
	printf("%s\n\n", device_properties.name);

	printf("Generiranje matrice ");

	CUT_SAFE_CALL(cutCreateTimer(&t));
	CUT_SAFE_CALL(cutStartTimer(t));
	
	//loadMatrix(m_in, "matrice/po32.mat", size);
	init(v, size * size);
	standard(v, v, m_in, size);

	CUT_SAFE_CALL(cutStopTimer(t));

	printf("%f\n", cutGetTimerValue(t));

	printf("CPU racuna ");

	CUT_SAFE_CALL(cutCreateTimer(&ta));
	CUT_SAFE_CALL(cutStartTimer(ta));
	
	//loadMatrix(m_in, "matrice/po32.mat", size);

	cpu_dpotrf(m_in, cpu_rez, size);
	CUT_SAFE_CALL(cutStopTimer(ta));

	printf("%f\n", cutGetTimerValue(ta));
	// GPU //
	int n = size;
	saveMatrix(m_in, "m.mat", n);
	saveMatrix(cpu_rez, "cpu_rez.mat", n);
	dim3 blokovaPoGridu, thredovaPoBloku;
	
	thredovaPoBloku.x = 16;
	thredovaPoBloku.y = 16;

	cudaMalloc((void **) &device_m, n * n * sizeof(float));
	cudaMalloc((void **) &device_m_out, n * n * sizeof(float));
	cudaMalloc((void **) &device_eye, 16 * 16 * sizeof(float));

	cudaMemset(device_m_out, 0, n * n *sizeof(float));
	

	printf("Kopiranje matrice na GPU: ");

	CUT_SAFE_CALL(cutCreateTimer(&t2));
	CUT_SAFE_CALL(cutStartTimer(t2));

	cudaMemcpy( device_m, 
				m_in, 
				n * n * sizeof(float), 
				cudaMemcpyHostToDevice );

/*	cudaMemcpy( device_eye, 
				eye, 
				16 * 16 * sizeof(float), 
				cudaMemcpyHostToDevice );*/

	CUT_SAFE_CALL(cutStopTimer(t2));

	printf("%f\n", cutGetTimerValue(t2));

	printf("GPU racuna: ");

	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutCreateTimer(&timer2));
	CUT_SAFE_CALL(cutStartTimer(timer2));

	int i;
	int it = n / 16 - 1;
	gpu_dpotrf<<< 1, 
				  thredovaPoBloku, 4*16*16*sizeof(float)
				   >>>
				  ( device_m, size, 0 );
	cudaThreadSynchronize();

	for (i = 0; i < n / 16 - 1; i++) {
		blokovaPoGridu.x = it;
		blokovaPoGridu.y = it;
		gpu_inv_l<<<it, 16, 4*16*16*sizeof(float)>>>(device_m, size, i);
	cudaThreadSynchronize();
//		gpu_mm_r<<<it, thredovaPoBloku>>>
//			(device_eye, device_m, size, i);
		if(it % 2){
			blokovaPoGridu.y = (it+1)/2;
			blokovaPoGridu.x = it;
		}
		else{
			blokovaPoGridu.y = it/2;
			blokovaPoGridu.x = it+1;
		}
	//	blokovaPoGridu.x = 1;
	//		blokovaPoGridu.y = 1;
	//blokovaPoGridu.y = it;
	//blokovaPoGridu.x = it;
		gpu_mm_a<<<blokovaPoGridu, thredovaPoBloku, 4*16*16*sizeof(float)>>>
		(device_m, size, i, it);
	cudaThreadSynchronize();
		gpu_dpotrf<<<1, thredovaPoBloku, 4*16*16*sizeof(float)>>>
		(device_m, size, i + 1);
	cudaThreadSynchronize();
		it--;
		
	}
	/*gpu_dpotrf<<< 1, thredovaPoBloku>>> ( device_m, size, 0 );


	cudaMemcpy( device_eye, eye, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice );
	gpu_inv_l<<<1, 16>>>(device_m, device_eye, size, 0);
	gpu_mm_r<<<it, thredovaPoBloku>>> (device_eye, device_m, size, 0);
	blokovaPoGridu.y = it;
	blokovaPoGridu.x = it;
	gpu_mm_a<<<blokovaPoGridu, thredovaPoBloku>>>(device_m, size, 0, it);
	gpu_dpotrf<<<1, thredovaPoBloku>>>	(device_m, size, 0 + 1);
	it--;
	cudaMemcpy( device_eye, eye, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice );
	gpu_inv_l<<<1, 16>>>(device_m, device_eye, size, 1);
	gpu_mm_r<<<it, thredovaPoBloku>>> (device_eye, device_m, size, 1);
	blokovaPoGridu.y = it;
	blokovaPoGridu.x = it;
	gpu_mm_a<<<blokovaPoGridu, thredovaPoBloku, 3 * 16 * 16 * sizeof(float)>>>(device_m, size, 1, it);
	gpu_dpotrf<<<1, thredovaPoBloku, 16 * 16 * sizeof(float)>>>	(device_m, size, 1 + 1);
	it--;
	cudaMemcpy( device_eye, eye, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice );
	gpu_inv_l<<<1, 16>>>(device_m, device_eye, size, 2);
	gpu_mm_r<<<it, thredovaPoBloku, 3 * 16 * 16 * sizeof(float)>>> (device_eye, device_m, size, 2);
	blokovaPoGridu.y = it;
	blokovaPoGridu.x = it;
	gpu_mm_a<<<blokovaPoGridu, thredovaPoBloku, 3 * 16 * 16 * sizeof(float)>>>(device_m, size, 2, it);
	gpu_dpotrf<<<1, thredovaPoBloku, 16 * 16 * sizeof(float)>>>	(device_m, size, 2 + 1);*/


	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(timer2));

	printf("%f\n", cutGetTimerValue(timer2));

	
	cudaMemcpy(m_out, device_m, 
			n * n * sizeof(float), cudaMemcpyDeviceToHost);
	printf("CPU %.20lf\n", cpu_rez[(size-1)*size+size-1]);
	printf("GPU %.20lf\n", m_out[(size-1)*size+size-1]);

	saveMatrix(m_out, "rez.mat", n);


	free(m_in);
	free(m_out);
	free(eye);
	cudaFree(device_m);
	cudaFree(device_m_out);
	cudaFree(device_eye);

	return 0;
}
