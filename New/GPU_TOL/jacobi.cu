/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define FULL_MASK 0xffffffff

__inline__ __device__
double warpReduceSum(double value) {
    for (unsigned int i = 16; i > 0; i /= 2) {
        value += __shfl_down_sync(FULL_MASK, value, i);
    }
    return value;
} 

__inline__ __device__
double blockReduceSum(double value) {
    __shared__ double smem[32]; // Max 32 warp sums

    if (threadIdx.x < warpSize) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    value = warpReduceSum(value);

    if (threadIdx.x % warpSize == 0) {
        smem[threadIdx.x / warpSize] = value;
    }
    __syncthreads();

    if (threadIdx.x < warpSize) {
        value = smem[threadIdx.x];
    }

    return warpReduceSum(value);
} 

__global__ void 
reduction_presum (double *a, int n, double *res)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double value = 0;

    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        value += a[i];
    }

    value = idx < n ? value : 0;
    value = blockReduceSum(value);

    if (threadIdx.x == 0) {
        atomicAdd(res, value);
    }
} 

__global__ void 
jacobi_iteration(double *u1, double *u2, double *f, double *res, int N, int NxN, int Nm1, float div6) 
{
    int i, j, k;
    double tmp=0.0;
    k = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    i = 1 + threadIdx.z + blockIdx.z * blockDim.z;

    if (i < Nm1 && j < Nm1 && k < Nm1) {
        u2[i*NxN + j*N + k] = div6 * ( \
        u1[(i-1)*NxN + j*N + k] + \
        u1[(i+1)*NxN + j*N + k] + \
        u1[i*NxN + (j-1)*N + k] + \
        u1[i*NxN + (j+1)*N + k] + \
        u1[i*NxN + j*N + (k-1)] + \
        u1[i*NxN + j*N + (k+1)] + \
        f[i*NxN + j*N + k] );

        // u2[i*NxN + j*N + k] = tmp;
        tmp = u1[i*NxN + j*N + k] - u2[i*NxN + j*N + k];
        res[i*NxN + j*N + k] = tmp * tmp;
    }
}


__global__ void print_arr(double *ptr, int N) {
    double sum=0.0;

    for (int i=0; i<N*N*N; i++) {
        if (ptr[i] > 10){
            printf("i: %d\t",i);
        }
    }

    printf("sum= %lf\n",sum);

    // for (int i=N; i<2*N; i++) {
    //     printf("elem: %lf\t",ptr[i]);
    // }
}

__global__ void print_ptr(double *ptr) {
    printf("ptr: %lf\n",*ptr);
}


void
jacobi(double *u1, double *u2, double *f, int N, int max_iter, double tolerance)
{
    // initialize variables
    int Nm1 = N - 1;
    int NxN = N * N;
    int iter = 0;

    double div6 = 1.0 / 6.0;
    double max_tol_squared = tolerance * tolerance;
    double h_tol_squared = max_tol_squared * 2;

    double *h_tol_ptr = &h_tol_squared;
    double *u3;
    
    double *part_tol;
    double *d_tol_ptr;
    cudaMalloc((void**)&part_tol, N*N*N*sizeof(double));
    cudaMalloc((void**)&d_tol_ptr, sizeof(double));

    // int d = 8;
    // dim3 dimGrid((N-2)/d, (N-2)/d, (N-2)/d);
    // dim3 dimBlock(d, d, d);

    int a, b, c;
    a = 32;
    b = 8;
    c = 4;
    dim3 dimGrid((N+a-1)/a, (N+b-1)/b, (N+c-1)/c);
    dim3 dimBlock(a, b, c);

    printf("max_tol: %lf \n",max_tol_squared);

    while (iter < max_iter && h_tol_squared > max_tol_squared) {
        // reset tolerance
        *h_tol_ptr = 0.0;
        cudaMemcpy(d_tol_ptr, h_tol_ptr, sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        
        // run jacobi iteration
        jacobi_iteration<<<dimGrid, dimBlock>>>(u1, u2, f, part_tol, N, NxN, Nm1, div6);
        cudaDeviceSynchronize();
        
        //calc new tolerance
        reduction_presum<<<(N*N*N+1023)/1024,1024>>>(part_tol, N*N*N, d_tol_ptr);
        cudaDeviceSynchronize();

        cudaMemcpy(h_tol_ptr, d_tol_ptr, sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // cudaDeviceSynchronize();
        // if (iter % 100 == 0){
        //     // print_arr<<<1,1>>>(part_tol,N);
        //     // printf("tol: %lf\n", h_tol_squared);
        //     printf("tol: %lf\n", *h_tol_ptr);
        // }

        // swap pointers
        u3 = u1; 
        u1 = u2;
        u2 = u3;

        // increment iteration
        iter++;
    }
    printf("max_iter = %d",iter);
}

