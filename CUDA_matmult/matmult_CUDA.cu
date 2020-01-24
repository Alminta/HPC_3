#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "kernfuncs.h"
#include "cublas_v2.h"

extern"C"{

#include <cblas.h>
    
void matmult_lib(int m, int n, int k, double *A, double *B, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
}


void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    const double alpha=1.0,beta=0.0;

    double *d_A,*d_B,*d_C;

    cudaMalloc( (void**)&d_A , m * k * sizeof(double));
    cudaMalloc( (void**)&d_B , n * k * sizeof(double));
    cudaMalloc( (void**)&d_C , m * n * sizeof(double));

    cudaMemcpy(d_A,A,m*k*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,n*k*sizeof(double),cudaMemcpyHostToDevice);

    cublasDgemm(handle, transa, transb, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n);

    cudaMemcpy(C,d_C,m*n*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C)
{
    
    double *d_A,*d_B,*d_C;
    
    cudaMalloc( (void**)&d_A , m * k * sizeof(double));
    cudaMalloc( (void**)&d_B , n * k * sizeof(double));
    cudaMalloc( (void**)&d_C , m * n * sizeof(double));
    
    cudaMemcpy(d_A,A,m*k*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,n*k*sizeof(double),cudaMemcpyHostToDevice);

    matmult_gpu1_kern <<<1,1>>>(m,n,k,d_A,d_B,d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C,d_C,m*n*sizeof(double),cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C)
{
    int bsize = 32;
    double *d_A,*d_B,*d_C;
    
    cudaMalloc( (void**)&d_A , m * k * sizeof(double));
    cudaMalloc( (void**)&d_B , n * k * sizeof(double));
    cudaMalloc( (void**)&d_C , m * n * sizeof(double));
    
    cudaMemcpy(d_A,A,m*k*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,n*k*sizeof(double),cudaMemcpyHostToDevice);

    dim3 dimGrid((n+bsize-1)/bsize,(m+bsize-1)/bsize);
    dim3 dimBlock(bsize,bsize);
    matmult_gpu2_kern <<<dimGrid,dimBlock>>>(m,n,k,d_A,d_B,d_C);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(C,d_C,m*n*sizeof(double),cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C)
{
    int bsize = 32,R = 2;
    double *d_A,*d_B,*d_C;
    
    cudaMalloc( (void**)&d_A , m * k * sizeof(double));
    cudaMalloc( (void**)&d_B , n * k * sizeof(double));
    cudaMalloc( (void**)&d_C , m * n * sizeof(double));
    
    cudaMemcpy(d_A,A,m*k*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,n*k*sizeof(double),cudaMemcpyHostToDevice);

    dim3 dimGrid((n + bsize - 1)/bsize,((m+R-1)/R+bsize-1)/bsize);
    dim3 dimBlock(bsize,bsize);
    matmult_gpu3_kern <<<dimGrid,dimBlock>>>(m,n,k,d_A,d_B,d_C);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(C,d_C,m*n*sizeof(double),cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C)
{
    int bsize = 32, R=8;
    double *d_A,*d_B,*d_C;
    
    cudaMalloc( (void**)&d_A , m * k * sizeof(double));
    cudaMalloc( (void**)&d_B , n * k * sizeof(double));
    cudaMalloc( (void**)&d_C , m * n * sizeof(double));
    
    cudaMemcpy(d_A,A,m*k*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,n*k*sizeof(double),cudaMemcpyHostToDevice);

    dim3 dimGrid((n + bsize - 1)/bsize,((m+R-1)/R+bsize-1)/bsize);
    dim3 dimBlock(bsize,bsize);
    matmult_gpu4_kern <<<dimGrid,dimBlock>>>(m,n,k,d_A,d_B,d_C);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(C,d_C,m*n*sizeof(double),cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

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

#ifndef __BLOCKSIZE
#define BLOCK_SIZE 16
#endif


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice);
        Matrix d_B;
        d_B.width = d_B.stride = B.width; d_B.height = B.height;
        size = B.width * B.height * sizeof(double);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size,
            cudaMemcpyHostToDevice);

            // Allocate C in device memory
            Matrix d_C;
            d_C.width = d_C.stride = C.width; d_C.height = C.height;
            size = C.width * C.height * sizeof(double);
            cudaMalloc(&d_C.elements, size);
            
            // Invoke kernel
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
            MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
            checkCudaErrors(cudaDeviceSynchronize());
            // Read C from device memory
            cudaMemcpy(C.elements, d_C.elements, size,
                cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C){
    Matrix MA,MB,MC;
    MB.width=MC.width=n;
    MA.width=MB.height=k;
    MA.height=MC.height=m;
    MA.elements = A;
    MB.elements = B;
    MC.elements = C;
    MatMul(MA,MB,MC);

}

}
