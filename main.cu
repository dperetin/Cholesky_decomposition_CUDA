#include <iostream>
#include <fstream>
#include <stdlib.h>
//#include <cutil.h>
#include <math.h>
#include <hdf5.h>
#include <string.h>
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

__global__ void GPU_SPOTRF(double *m, int size, int p)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double a[16][16+1];
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

    if (ty >= tx) 
        m[(tx + 16 * p) * size + ty + 16 * p] = a[ty][tx];
}

__global__ void GPU_STRSM(double *m, int size, int p)
{
    int i, j;

    int tid = threadIdx.x;
    int bx = blockIdx.x + 1;

    __shared__ double b[16][16];

    for(i = 0; i < 16; i++) {
        b[i][tid] = m[(i + p * 16) * size + tid + (bx + p) * 16];
    }

    b[0][tid] = b[0][tid] / m[(0 + p * 16) * size + (0 + 16 * p)];

    for (i = 1; i < 16; i++) {
        //double d = 0;
        for (j = 0; j < i; j++) {
             b[i][tid] -=  m[(j + p * 16) * size + (i + p * 16)] * b[j][tid];
        }
        b[i][tid] = (b[i][tid]) / m[(i + p * 16) * size + (i + 16 *p)];
    }
    __syncthreads();
    for(i = 0; i < 16; i++) {
        m[(i + p * 16) * size + tid + (bx + p) * 16] = b[i][tid];
    }
}

__global__ void GPU_SGEMM(double *m, int size, int p, int s, int mod, int visina)
{
    __shared__ double s_a1[16][16];
    __shared__ double s_a2[16][16];
    __shared__ double s_a3[16][16];
    __shared__ double s_b1[16][16];
    __shared__ double s_b2[16][16];
    __shared__ double s_b3[16][16];
    
    
    double s_c1 = 0, s_c2 = 0, s_c3 = 0, s_c4 = 0, s_c5 = 0, s_c6 = 0, s_c7 = 0, s_c8 = 0, s_c9 = 0;
    int tx = threadIdx.x, i;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    
    int z1 = (ty + p * 16) * size + tx + (s) * 16;
    int z2 = (ty + p * 16) * size + tx + (s + bx * 3) * 16;
    int z3 = (ty + (s) * 16) * size + tx + (s + bx * 3) * 16;
    int z4 = (ty + (s+1) * 16) * size + tx + (s + (bx * 3)) * 16;
    int z5 = (ty + (s+2) * 16) * size + tx + (s + (bx * 3)) * 16;
    if (bx + 1 == gridDim.x) {
        if (mod == 0) 
            return;
        
        if (mod == 1 && visina == 1) {
        
            s_a1[ty][tx] = m[z1];
            s_b1[ty][tx] = m[z2];
            __syncthreads();
           // #pragma unroll 16
            for (i = 0; i < 16; i++)
            {
                s_c1 += s_a1[i][ty] * s_b1[i][tx];
                
            }
            m[z3] -= s_c1;
            return;
        }
        if ( mod == 2 && visina == 2) {
        
            s_a1[ty][tx] = m[z1];
            s_a2[ty][tx] = m[z1+16];
            s_b1[ty][tx] = m[z2];
            s_b2[ty][tx] = m[z2+16];
            __syncthreads();
            #pragma unroll 16
            for (i = 0; i < 16; i++)
            {
                s_c1 += s_a1[i][ty] * s_b1[i][tx];
                s_c2 += s_a1[i][ty] * s_b2[i][tx];
                s_c4 += s_a2[i][ty] * s_b1[i][tx];
                s_c5 += s_a2[i][ty] * s_b2[i][tx];
                
                
            }
            m[z3] -= s_c1;
            m[z3+16] -= s_c2;
            m[z4] -= s_c4;
            m[z4+16] -= s_c5;
            return;
        }
        if ( mod == 1 && visina >= 3) {
        
            s_a1[ty][tx] = m[z1];
            s_a2[ty][tx] = m[z1+16];
            s_a3[ty][tx] = m[z1+32];
            s_b1[ty][tx] = m[z2];
            
            __syncthreads();
            #pragma unroll 16
            for (i = 0; i < 16; i++)
            {
                s_c1 += s_a1[i][ty] * s_b1[i][tx];      
                s_c4 += s_a2[i][ty] * s_b1[i][tx];
                s_c7 += s_a3[i][ty] * s_b1[i][tx];
                
                
                
            }
            m[z3] -= s_c1;
            m[z4] -= s_c4;
            m[z5] -= s_c7;
            return;
        }
    }

    s_a1[ty][tx] = m[z1];
    s_a2[ty][tx] = m[z1+16];
    s_a3[ty][tx] = m[z1+32];
    s_b1[ty][tx] = m[z2];
    s_b2[ty][tx] = m[z2+16];
    s_b3[ty][tx] = m[z2+32];
    
    __syncthreads();

    #pragma unroll 16
    for (i = 0; i < 16; i++)
    {
        s_c1 += s_a1[i][ty] * s_b1[i][tx];
        s_c2 += s_a1[i][ty] * s_b2[i][tx];
        s_c3 += s_a1[i][ty] * s_b3[i][tx];
        s_c4 += s_a2[i][ty] * s_b1[i][tx];
        s_c5 += s_a2[i][ty] * s_b2[i][tx];
        s_c6 += s_a2[i][ty] * s_b3[i][tx];
        s_c7 += s_a3[i][ty] * s_b1[i][tx];
        s_c8 += s_a3[i][ty] * s_b2[i][tx];
        s_c9 += s_a3[i][ty] * s_b3[i][tx];
    }
    
    
    m[z3] -= s_c1;
    m[z3+16] -= s_c2;
    m[z3+32] -= s_c3;
    m[z4] -= s_c4;
    m[z4+16] -= s_c5;
    m[z4+32] -= s_c6;
    m[z5] -= s_c7;
    m[z5+ 16] -= s_c8;
    m[z5+32] -= s_c9;
    
    
}

int main(int argc, char *argv[])
{

    if (argc < 2 || argc > 4)
    {
        fprintf(stderr, "GPUCHOL [red matrice] [opcionalno - ime datoteke]");
        return 1;
    }

    int size = atoi(argv[1]);

    hid_t       file_id, dataset_id, dataspace_id;
    
    double *m_in, *device_m, *v, *cpu_rez;
    m_in = new double[size * size];
//  m_out = new double[size * size];
    
    memset(m_in, 0, size * size * sizeof(double));
//  memset(m_out, 0, size * size * sizeof(double));
    
    

    int deviceOrdinal = 0;
    cudaSetDevice(deviceOrdinal);
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, deviceOrdinal);
    printf("\n%s\n\n", device_properties.name);


    if (argc == 2) {

        cpu_rez = new double[size * size];
        v = new double[size * size];
        memset(cpu_rez, 0, size * size * sizeof(double));

 
        init(v, size * size);
        standard(v, v, m_in, size);


    
        cpu_potrf(m_in, cpu_rez, size);

    }

    if (argc == 3) {

        printf("Ucitavanje matrice iz datoteke:\t");
        fflush(stdout);

//      CUT_SAFE_CALL(cutCreateTimer(&t_mat_load));
//      CUT_SAFE_CALL(cutStartTimer(t_mat_load));
    
        file_id = H5Fopen(argv[2], H5F_ACC_RDWR, H5P_DEFAULT);
        dataset_id = H5Dopen(file_id, "/16", H5P_DEFAULT);
        H5Dread(dataset_id, H5T_IEEE_F64LE, 
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, m_in);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
//      CUT_SAFE_CALL(cutStopTimer(t_mat_load));

//      printf("%f\n\n", cutGetTimerValue(t_mat_load));

    }

    cudaStream_t stream0;
    cudaStreamCreate(&stream0);
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // GPU //
    int n = size;
    cudaEvent_t start, stop;
    dim3 blokovaPoGridu, thredovaPoBloku;
    
    thredovaPoBloku.x = 16;
    thredovaPoBloku.y = 16;

    cudaMalloc((void **) &device_m, n * n * sizeof(double));

    

//  CUT_SAFE_CALL(cutCreateTimer(&t_h2d));
//  CUT_SAFE_CALL(cutStartTimer(t_h2d));
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemcpy(device_m, m_in, n * n * sizeof(double), cudaMemcpyHostToDevice);


//  CUT_SAFE_CALL(cutStopTimer(t_h2d));

//  printf("%f\n", cutGetTimerValue(t_h2d));

    


//  CUT_SAFE_CALL(cutCreateTimer(&t_gpu));
//  CUT_SAFE_CALL(cutStartTimer(t_gpu));
    
    int i, j;
    int it = n / 16 - 1;
    GPU_SPOTRF <<<1, thredovaPoBloku>>> (device_m, size, 0);

    for (i = 0; i < n / 16 - 1; i++) {

        GPU_STRSM <<<it, 16>>> (device_m, size, i);
        
        for (j = i; j < n / 16 - 1; j += 6){
            //printf("\n%d %d %d\n",(n / 16 - 1 - j) / 3 + 1, (n / 16 - 1 - j) % 3, n/16 - (j+1));
            GPU_SGEMM <<<(n / 16 - 1 - j) / 3 + 1, thredovaPoBloku, 16*16*16, stream0>>> 
                (device_m, size, i, j + 1, (n / 16 - 1 - j) % 3, n/16 - (j+1));
            if (j+3 < n / 16 - 1)    
            GPU_SGEMM <<<(n / 16 - 1 - j-3) / 3 + 1, thredovaPoBloku, 16*16*16, stream1>>> 
                (device_m, size, i, j+3 + 1, (n / 16 - 1 - j-3) % 3, n/16 - (j+3+1));
            
        }
    
        GPU_SPOTRF <<<1, thredovaPoBloku>>> (device_m, size, i + 1);
    
        it--;
    }
    


//  CUT_SAFE_CALL(cutStopTimer(t_gpu));

//  printf("%f\n", cutGetTimerValue(t_gpu));
    

   
//  CUT_SAFE_CALL(cutCreateTimer(&t_d2h));
//  CUT_SAFE_CALL(cutStartTimer(t_d2h));

    cudaMemcpy(m_in, device_m, 
            n * n * sizeof(double), cudaMemcpyDeviceToHost);

    
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

    hsize_t     dims[2];
    dims[0] = dims[1] = size;
    
    
    if (argc > 2) { 
        char str[80];
        puts(argv[2]);
        strcpy (str,"rez");
        strcat (str, argv[2]);
puts(str);

        file_id = H5Fcreate(str, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    //file_id = H5Fopen("rez.h5", H5F_ACC_RDWR, H5P_DEFAULT);
        dataspace_id = H5Screate_simple(2, dims, NULL);
        dataset_id = H5Dcreate(file_id, "/16", H5T_IEEE_F64LE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        H5Dwrite(dataset_id, H5T_IEEE_F64LE, 
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, m_in);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
    }
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

