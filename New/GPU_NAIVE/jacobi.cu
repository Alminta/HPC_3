/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stdio.h>
#include <math.h>
#include <cuda.h>



__global__ void 
jacobi_iteration(double *u1, double *u2, double *f, int N, int NxN, int Nm1) 
{
    int i, j, k;
    const double div6 = 1.0 / 6.0;

    k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    i = threadIdx.z + blockIdx.z * blockDim.z + 1;

    // if (i > 30 || j > 30 || k > 30){
    //     printf("%d, %d, %d\n",i,j,k);
    // }
    // printf("%d, %d, %d\n",i,j,k);
    // 0 < i && i < Nm1 && 0 < j && j < Nm1 && 0 < k && k < Nm1
    if (i < Nm1 && j < Nm1 && k < Nm1) {
        u2[i*NxN + j*N + k] = div6 * ( \
        u1[(i-1)*NxN + j*N + k] + \
        u1[(i+1)*NxN + j*N + k] + \
        u1[i*NxN + (j-1)*N + k] + \
        u1[i*NxN + (j+1)*N + k] + \
        u1[i*NxN + j*N + (k-1)] + \
        u1[i*NxN + j*N + (k+1)] + \
        f[i*NxN + j*N + k] );
    }
}

void
jacobi(double *u1, double *u2, double *f, int N, int max_iter, double tolerance)
{
    // initialize variables
    int Nm1 = N - 1;
    int NxN = N * N;
    int iter = 0;

    double div6 = 1.0 / 6.0;
    double *u3;
    
    int a, b, c;
    a = 32;
    b = 8;
    c = 4;


    dim3 dimGrid((N+a+1)/a, (N+b-1)/b, (N+c-1)/c);
    dim3 dimBlock(a,b,c);


    while (iter < max_iter) {

        // run jacobi iteration
        jacobi_iteration<<<dimGrid, dimBlock>>>(u1, u2, f, N, NxN, Nm1);
        cudaDeviceSynchronize();

        // swap pointers
        u3 = u1; 
        u1 = u2;
        u2 = u3;

        // increment iteration
        iter++;
    }
}

