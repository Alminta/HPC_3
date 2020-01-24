#ifndef __KERNFUNCS_H
#define __KERNFUNCS_H

#ifndef __MATRIX
#define __MATRIX
typedef struct
{
    int width;
    int height;
    int stride;
    double *elements;
} Matrix;
#endif

__global__ void matmult_gpu1_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C);
__global__ void matmult_gpu2_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C);
__global__ void matmult_gpu3_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C);
__global__ void matmult_gpu4_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C);
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
#endif
