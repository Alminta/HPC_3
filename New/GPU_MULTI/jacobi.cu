/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stdio.h>
#include <math.h>
#include <cuda.h>

__global__ void 
jacobi_iteration_d0(double *d0_u1, double *d0_u2, double *d0_f, double *d1_u1, double *d1_u2, double *d1_f, int N, int NxN, int Nm1, int Ndiv2, float div6) 
{
    int i, j, k;

    k = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    i = 1 + threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i < Ndiv2-1 && j < Nm1 && k < Nm1) { // if no peer access 
        d0_u2[i*NxN + j*N + k] = div6 * ( \
            d0_u1[(i-1)*NxN + j*N + k] + \
            d0_u1[(i+1)*NxN + j*N + k] + \
            d0_u1[i*NxN + (j-1)*N + k] + \
            d0_u1[i*NxN + (j+1)*N + k] + \
            d0_u1[i*NxN + j*N + (k-1)] + \
            d0_u1[i*NxN + j*N + (k+1)] + \
            d0_f[i*NxN + j*N + k] ); 
    }
    else if ( i == Ndiv2-1 && j < Nm1 && k < Nm1) { // if peer access get element from top (i=0) of d1_u1
        // printf("%d, %d, %d\n",i,j,k);
        d0_u2[i*NxN + j*N + k] = div6 * ( \
            d0_u1[(i-1)*NxN + j*N + k] + \
            d1_u1[j*N + k] + \
            d0_u1[i*NxN + (j-1)*N + k] + \
            d0_u1[i*NxN + (j+1)*N + k] + \
            d0_u1[i*NxN + j*N + (k-1)] + \
            d0_u1[i*NxN + j*N + (k+1)] + \
            d0_f[i*NxN + j*N + k] );
    }
}

__global__ void 
jacobi_iteration_d1(double *d0_u1, double *d0_u2, double *d0_f, double *d1_u1, double *d1_u2, double *d1_f, int N, int NxN, int Nm1, int Ndiv2, float div6) 
{
    int i, j, k;

    k = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;
    i = threadIdx.z + blockIdx.z * blockDim.z;

    if (i == 0 && j > 0 && j < Nm1 && k > 0 && k < Nm1) { // if peer access get element from bottom (i=N/2) of d0_u1
        d1_u2[i*NxN + j*N + k] = div6 * ( \
            d0_u1[(Ndiv2-1)*NxN + j*N + k] + \
            d1_u1[(i+1)*NxN + j*N + k] + \
            d1_u1[i*NxN + (j-1)*N + k] + \
            d1_u1[i*NxN + (j+1)*N + k] + \
            d1_u1[i*NxN + j*N + (k-1)] + \
            d1_u1[i*NxN + j*N + (k+1)] + \
            d1_f[i*NxN + j*N + k] );
    }
    else if (i < Ndiv2-1 && j > 0 && j < Nm1 && k > 0 && k < Nm1) { // if no peer access
        d1_u2[i*NxN + j*N + k] = div6 * ( \
            d1_u1[(i-1)*NxN + j*N + k] + \
            d1_u1[(i+1)*NxN + j*N + k] + \
            d1_u1[i*NxN + (j-1)*N + k] + \
            d1_u1[i*NxN + (j+1)*N + k] + \
            d1_u1[i*NxN + j*N + (k-1)] + \
            d1_u1[i*NxN + j*N + (k+1)] + \
            d1_f[i*NxN + j*N + k] );
    }
}


void
jacobi(double *h_u1, double *h_u2, double *h_f, int N, int max_iter, double tolerance)
{
    int total_size = sizeof(double) * N * N * N / 2;
    int total_elems = N * N * N / 2;

    cudaSetDevice(0);
    double 	*d0_u1 = NULL;
    double 	*d0_u2 = NULL;
    double 	*d0_f = NULL;
    cudaMalloc((void**)&d0_u1, total_size);
    cudaMalloc((void**)&d0_u2, total_size);
    cudaMalloc((void**)&d0_f, total_size);
    cudaMemcpy(d0_u1, h_u1, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d0_u2, h_u2, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d0_f, h_f, total_size, cudaMemcpyHostToDevice);


    cudaSetDevice(1);
    double 	*d1_u1 = NULL;
    double 	*d1_u2 = NULL;
    double 	*d1_f = NULL;
    cudaMalloc((void**)&d1_u1, total_size);
    cudaMalloc((void**)&d1_u2, total_size);
    cudaMalloc((void**)&d1_f, total_size);
    cudaMemcpy(d1_u1, h_u1 + total_elems, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d1_u2, h_u2 + total_elems, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d1_f, h_f + total_elems, total_size, cudaMemcpyHostToDevice);
  

    // initialize variables
    int Nm1 = N - 1;
    int NxN = N * N;
    int Ndiv2 = N / 2;
    int iter = 0;

    double div6 = 1.0 / 6.0;
    double *tmp_ptr;

    int a, b, c;
    a = 32;
    b = 8;
    c = 4;

    dim3 dimGrid((N+a+1)/a, (N+b-1)/b, (N+2*c-1)/(2*c));
    dim3 dimBlock(a,b,c);
    
    // dim3 dimGrid((N-2)/(2*d), (N-2)/(d), (N-2)/(d));
    // dim3 dimBlock(d, d, d);
    
    while (iter < max_iter) {

        // run jacobi iteration
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        jacobi_iteration_d0<<<dimGrid, dimBlock>>>(d0_u1, d0_u2, d0_f, d1_u1, d1_u2, d1_f, N, NxN, Nm1, Ndiv2, div6);

        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        jacobi_iteration_d1<<<dimGrid, dimBlock>>>(d0_u1, d0_u2, d0_f, d1_u1, d1_u2, d1_f, N, NxN, Nm1, Ndiv2, div6);

        cudaDeviceSynchronize();

        // cudaSetDevice(0);
        // cudaDeviceSynchronize();

        // swap pointers
        tmp_ptr = d0_u1; 
        d0_u1 = d0_u2;
        d0_u2 = tmp_ptr;

        tmp_ptr = d1_u1; 
        d1_u1 = d1_u2;
        d1_u2 = tmp_ptr;

        // increment iteration
        iter++;
    }

    // copy memory to CPU
    cudaMemcpy(h_u1, d0_u1, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_u1 + total_elems, d1_u1, total_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d0_u1);
    cudaFree(d0_u2);
    cudaFree(d0_f);
    cudaFree(d1_u1);
    cudaFree(d1_u2);
    cudaFree(d1_f);

}

