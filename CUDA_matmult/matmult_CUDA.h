#ifndef __MATMULT_CUDA_H
#define __MATMULT_CUDA_H

void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpu6(int m, int n, int k, double *A, double *B, double *C);
void matmult_lib(int m, int n, int k, double *A, double *B, double *C);
void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C);

#endif
