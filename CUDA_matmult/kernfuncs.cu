#include <math.h>
__global__ 
void matmult_gpu1_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C)
{
    int i, j, h;

    for(j = 0; j < n; j++){
        for(i = 0; i < m; i++){
            d_C[i*n + j] = 0;
        }
    }

    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            for(h = 0; h < k; h++){
                d_C[i*n + j] += d_A[i*k + h] * d_B[h*n + j];
            }
        }
    }
}


__global__ 
void matmult_gpu2_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C)
{

    int i, j, h;
    double tmp = 0;

    j = threadIdx.x + blockDim.x*blockIdx.x;
    i = threadIdx.y + blockDim.y*blockIdx.y;

    if (j<n&&i<m){
        for(h = 0; h < k; h++){
            tmp += d_A[i*k + h] * d_B[h*n + j];
        }
        d_C[i*n + j] = tmp;
    }

}


__global__ 
void matmult_gpu3_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C)
{

    int i, j, h;
    double tmp1 = 0,tmp2 = 0;
    double b;
    j = threadIdx.x + blockDim.x*blockIdx.x;
    i = threadIdx.y + blockDim.y*blockIdx.y;
    i=i*2;

    if (i<m&&j<n){
        if(1<(m-i)){
            for(h = 0; h < k; h++){
                b = d_B[h*n + j];
                tmp1 += d_A[i*k + h] * b;
                tmp2 += d_A[i*k + h+k] * b;
            }
            d_C[i*n + j] = tmp1;
            
            d_C[i*n + j+n] = tmp2;
        }else{
            for(h = 0; h < k; h++){
                tmp1 += d_A[i*k + h] * d_B[h*n + j];
            }
            d_C[i*n + j] = tmp1;
        }
    }

}


__global__
void matmult_gpu4_kern(int m, int n, int k, double *d_A, double *d_B, double *d_C)
{

    int i, j, h, r,R =8;
    double b,tmp[8]={0};

    j = threadIdx.x + blockDim.x*blockIdx.x;
    i = threadIdx.y + blockDim.y*blockIdx.y;
    i=i*R;
    
    if (i<m&&j<n){
        if (R<(m-i)){
            for(h = 0; h < k; h++){
                b = d_B[h*n + j];
                for (r=0;r<R;r++){
                    tmp[r] += d_A[(i+r)*k + h] * b;
                }
            }
            for (r=0;r<R;r++){
                d_C[(i+r)*n + j] = tmp[r];
                
            }
        }
        else{
            int rBreak = m-i;
            for(h = 0; h < k; h++){
                b = d_B[h*n + j];
                for (r=0;r<rBreak;r++){
                    tmp[r] += d_A[(i+r)*k + h] * b;
                }
            }
            for (r=0;r<rBreak;r++){
                d_C[(i+r)*n + j] = tmp[r];
                
            }
        }
            
    }

}

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
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

// Get a matrix element
__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           double value)
{
    A.elements[row * A.stride + col] = value;
}


// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Thread block size

// Forward declaration of the matrix multiplication kernel




// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}