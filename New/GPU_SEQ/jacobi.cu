#include <stdio.h>
#include <math.h>
#include <cuda.h>

__global__ void 
jacobi_iteration(double *u1, double *u2, double *f, int N, int NxN, int Nm1) 
{
    int i, j, k;
    const double div6 = 1.0 / 6.0;

    for (i = 1; i < Nm1; i++){
        for (j = 1; j < Nm1; j++){
            for (k = 1; k < Nm1; k++){
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
    }
}

void
jacobi(double *u1, double *u2, double *f, int N, int max_iter, double tolerance)
{
    // initialize variables
    int Nm1 = N - 1;
    int NxN = N * N;
    int iter = 0;

    double *u3;
    
    while (iter < max_iter) {
        // run jacobi iteration
        jacobi_iteration<<<1,1>>>(u1, u2, f, N, NxN, Nm1);
        cudaDeviceSynchronize();

        // swap pointers
        u3 = u1; 
        u1 = u2;
        u2 = u3;

        // increment iteration
        iter++;
    }
}

