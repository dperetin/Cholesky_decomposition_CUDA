#include <iostream>
#include <fstream>
#include <stdlib.h>
//#include <cutil.h>
#include <math.h>
#include <hdf5.h>

using namespace std;

void cpu_potrf(float *m_in, float *m_out, int size)
{
    for (int i = 0; i < size; i++) {
        float sum = 0;
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

void standard (float *A, float  *B, float *C, int size)
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

void init(float *v, int n)
{
    int i;
    srand(time(NULL));
    for (i = 0; i < n; i++)
        v[i] = rand() / (float(RAND_MAX) + 1) - 1; 
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

__global__ void gpu_potrf(float (*m)[512], int size, int p)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float a[16][16 + 1];
    a[ty][tx] = m[(ty + 16 * p)][tx + 16 * p];

    __syncthreads();

    float d;

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
        m[(tx + 16 * p) ][ty + 16 * p] = a[ty][tx];
    
}

__global__ void gpu_inv_l(float (*u)[512], int size, int p)
{
    int i, j;

    int tid = threadIdx.x;
    int bx = blockIdx.x + 1;

    __shared__ float b[16][16];

    for(i = 0; i < 16; i++)
        b[i][tid] = u[(i + p * 16) ][ tid + (bx + p) * 16];

    b[0][tid] = b[0][tid] / u[(0 + p * 16) ][ + (0 + 16 * p)];

    for (i = 1; i < 16; i++){
        for (j = 0; j < i; j++) {
            b[i][tid] = b[i][tid] - 
                u[(j + p * 16) ][ (i + p * 16)] * b[j][tid];
        }
        b[i][tid] = b[i][tid] / u[(i + p * 16) ][ + (i + 16 *p)];
    }

    for(i = 0; i < 16; i++)
        u[(i + p * 16) ][ tid + (bx + p) * 16] = b[i][tid];
}

__global__ void gpu_mm_a(float (*m)[512], int size, int p, int s, int mod, int visina)
{
    __shared__ float s_a1[16][16];
    __shared__ float s_a2[16][16];
    __shared__ float s_b1[16][16];
    __shared__ float s_b2[16][16];
    //__shared__ float s_c[16][16];
    float s_c1 = 0, s_c2 = 0, s_c3 = 0, s_c4 = 0;
    int tx = threadIdx.x, i;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    
    if (bx + 1 != gridDim.x)
    {
      s_a1[ty][tx] = m[(ty + p * 16) ][ tx + (s) * 16];
    s_a2[ty][tx] = m[(ty + p * 16) ][ tx + (s + 1) * 16];
    s_b1[ty][tx] = m[(ty + p * 16) ][ tx + (s + bx * 2) * 16];
    s_b2[ty][tx] = m[(ty + p * 16) ][ tx + (s + bx * 2 + 1) * 16];

    __syncthreads();

    #pragma unroll 16
    for (i = 0; i < 16; i++)
    {
        s_c1 += s_a1[i][ty] * s_b1[i][tx];
        s_c2 += s_a1[i][ty] * s_b2[i][tx];
        s_c3 += s_a2[i][ty] * s_b1[i][tx];
        s_c4 += s_a2[i][ty] * s_b2[i][tx];
    }
    
    
    m[(ty + (s) * 16) ][ tx + (s + bx * 2) * 16] -= s_c1;
    m[(ty + (s) * 16) ][ tx + (s + (bx *2)+ 1) * 16] -= s_c2;
    m[(ty + (s+1) * 16) ][ tx + (s + (bx * 2)) * 16] -= s_c3;
    m[(ty + (s+1) * 16) ][ tx + (s + (bx * 2)+ 1) * 16] -= s_c4;
    return;
    }
    if (bx + 1 == gridDim.x && mod == 1 && visina == 1) {
        s_a1[ty][tx] = m[(ty + p * 16) ][ tx + (s) * 16];
        s_b1[ty][tx] = m[(ty + p * 16) ][ tx + (s + bx * 2) * 16];
        __syncthreads();
        #pragma unroll 16
        for (i = 0; i < 16; i++)
        {
            s_c1 += s_a1[i][ty] * s_b1[i][tx];
            
        }
        m[(ty + (s) * 16) ][ tx + (s + bx * 2) * 16] -= s_c1;
        return;
    }
    if (bx + 1 == gridDim.x && mod == 0) 
        return;
    if (bx + 1 == gridDim.x && mod == 1) {
        s_a1[ty][tx] = m[(ty + p * 16) ][ tx + (s) * 16];
        s_a2[ty][tx] = m[(ty + p * 16) ][ tx + (s + 1) * 16];
        s_b1[ty][tx] = m[(ty + p * 16) ][ tx + (s + bx * 2) * 16];
        __syncthreads();
        #pragma unroll 16
        for (i = 0; i < 16; i++)
        {
            s_c1 += s_a1[i][ty] * s_b1[i][tx];
            s_c3 += s_a2[i][ty] * s_b1[i][tx];
            
        }
        m[(ty + (s) * 16) ][tx + (s + bx * 2) * 16] -= s_c1;
        m[(ty + (s+1) * 16) ][tx + (s + (bx * 2)) * 16] -= s_c3;
        return;
    }
    
    
}

int main(int argc, char *argv[])
{

    if (argc < 2 || argc > 4)
    {
        fprintf(stderr, "GPUCHOL [red matrice] [opcionalno - ime datoteke]");
        return 1;
    }

    int size = atoi(argv[1]);

    hid_t       file_id, dataset_id;
    
    float *m_in, (*device_m)[512], *v, *cpu_rez;
    m_in = new float[size * size];
//  m_out = new float[size * size];
    
    memset(m_in, 0, size * size * sizeof(float));
//  memset(m_out, 0, size * size * sizeof(float));
    
    

    int deviceOrdinal = 0;
    cudaSetDevice(deviceOrdinal);
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, deviceOrdinal);
    printf("\n%s\n\n", device_properties.name);


    if (argc == 2) {

        cpu_rez = new float[size * size];
        v = new float[size * size];
        memset(cpu_rez, 0, size * size * sizeof(float));

        printf("Generiranje matrice:\t\t");
        fflush(stdout);

//      CUT_SAFE_CALL(cutCreateTimer(&t_mat_gen));
//      CUT_SAFE_CALL(cutStartTimer(t_mat_gen));
    
        
        init(v, size * size);
        standard(v, v, m_in, size);

//      CUT_SAFE_CALL(cutStopTimer(t_mat_gen));

//      printf("%f\n", cutGetTimerValue(t_mat_gen));

        printf("CPU racuna:\t\t\t");

//      CUT_SAFE_CALL(cutCreateTimer(&t_cpu));
//      CUT_SAFE_CALL(cutStartTimer(t_cpu));
    
        cpu_potrf(m_in, cpu_rez, size);
//      CUT_SAFE_CALL(cutStopTimer(t_cpu));

//      printf("%f\n\n", cutGetTimerValue(t_cpu));

//      saveMatrix(m_in, "m.mat", size);
//      saveMatrix(cpu_rez, "cpu_rez.mat", size);
    }

    if (argc == 3) {

        printf("Ucitavanje matrice iz datoteke:\t");
        fflush(stdout);

//      CUT_SAFE_CALL(cutCreateTimer(&t_mat_load));
//      CUT_SAFE_CALL(cutStartTimer(t_mat_load));
    
        file_id = H5Fopen(argv[2], H5F_ACC_RDWR, H5P_DEFAULT);
        dataset_id = H5Dopen(file_id, "/16", H5P_DEFAULT);
        H5Dread(dataset_id, H5T_IEEE_F32LE, 
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, m_in);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
//      CUT_SAFE_CALL(cutStopTimer(t_mat_load));

//      printf("%f\n\n", cutGetTimerValue(t_mat_load));

    }

    // GPU //
    int n = size;
    cudaEvent_t start, stop;
    dim3 blokovaPoGridu, thredovaPoBloku;
    
    thredovaPoBloku.x = 16;
    thredovaPoBloku.y = 16;

    cudaMalloc((void **) &device_m, n * n * sizeof(float));

    printf("Kopiranje matrice na GPU:\t");

//  CUT_SAFE_CALL(cutCreateTimer(&t_h2d));
//  CUT_SAFE_CALL(cutStartTimer(t_h2d));
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemcpy(device_m, m_in, n * n * sizeof(float), cudaMemcpyHostToDevice);


//  CUT_SAFE_CALL(cutStopTimer(t_h2d));

//  printf("%f\n", cutGetTimerValue(t_h2d));

    printf("GPU racuna:\t\t\t");



//  CUT_SAFE_CALL(cutCreateTimer(&t_gpu));
//  CUT_SAFE_CALL(cutStartTimer(t_gpu));

    int i, j;
    int it = n / 16 - 1;
    gpu_potrf <<<1, thredovaPoBloku>>> (device_m, size, 0);

    for (i = 0; i < n / 16 - 1; i++) {

        gpu_inv_l <<<it, 16>>> (device_m, size, i);
        
        for (j = i; j < n / 16 - 1; j += 2)
        
            gpu_mm_a <<<(n / 16 - 1 - j) / 2 + 1, thredovaPoBloku>>> 
                (device_m, size, i, j + 1, (n / 16 - 1 - j) % 2, n/16 - (j+1));
    
        gpu_potrf <<<1, thredovaPoBloku>>> (device_m, size, i + 1);
    
        it--;
    }
    


//  CUT_SAFE_CALL(cutStopTimer(t_gpu));

//  printf("%f\n", cutGetTimerValue(t_gpu));
    

    printf("Kopiranje matrice natrag:\t");
//  CUT_SAFE_CALL(cutCreateTimer(&t_d2h));
//  CUT_SAFE_CALL(cutStartTimer(t_d2h));

    cudaMemcpy(m_in, device_m, 
            n * n * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float vrijeme;
    cudaEventElapsedTime(&vrijeme, start, stop);
    
//  CUT_SAFE_CALL(cutStopTimer(t_d2h));

//  printf("%f\n", cutGetTimerValue(t_d2h));
//  printf("----------------------------------------------\n");
//  printf("UKUPNO:\t\t\t\t%f\n\n", cutGetTimerValue(t_d2h) + 
//                         cutGetTimerValue(t_h2d) + 
//                         cutGetTimerValue(t_gpu));
    printf("\nUKUPNO: %f\n", vrijeme);
    if (argc == 2) {
        printf("CPU %.13f\n", cpu_rez[(size - 1) * size + size - 1]);
    }
    printf("GPU %.13f\n", m_in[(size - 1) * size + size - 1]);


//  saveMatrix(m_out, "rez.mat", n);

    free(m_in);
//  free(m_out);
    if (argc == 2) {
        free(cpu_rez);
        free(v);
    }
    cudaFree(device_m);

    return 0;
}

