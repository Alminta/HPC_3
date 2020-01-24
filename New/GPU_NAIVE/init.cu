#include <math.h>
#include <stdio.h>


void
zeros(double *m, int N)
{
    int i,j,k;
    int ind_i, ind_j;
    int NxN = N * N;

    for (i = 0; i < N; i++){
        ind_i = i*NxN;
        for (j = 0; j < N; j++){
            ind_j = j*N;
            for (k = 0; k < N; k++){
                m[ind_i + ind_j + k] = 0.0;
            }
        }
    }
}

void
full(double *m, int N, double start_T)
{
    int i,j,k;
    int ind_i, ind_j;
    int NxN = N * N;

    for (i = 0; i < N; i++){
        ind_i = i*NxN;
        for (j = 0; j < N; j++){
            ind_j = j*N;
            for (k = 0; k < N; k++){
                m[ind_i + ind_j + k] = start_T;
            }
        }
    }
}


void
boundary(double *m, int N)
{
    int i, j, k;
    int ind_i, ind_j, ind_j2;
    int Nm1 = N - 1;
    int NxN = N * N;

    for (i = 0; i < N; i++){
        ind_i = i*NxN;
        for (k = 0; k < N; k++){
            m[ind_i + k] = 0.0;
        }
    }

    for (i = 0; i < N; i++){
        ind_i = i*NxN + Nm1*N;
        for (k = 0; k < N; k++){
            m[ind_i + k] = 20.0;
        }
    }

    for (j = 0; j < N; j++){
        ind_j = j * N;
        ind_j2 = Nm1*NxN + ind_j;
        for (k = 0; k < N; k++){
            m[ind_j + k] = 20.0;
            m[ind_j2 + k] = 20.0;
        }
    }

    for (i = 0; i < N; i++){
        ind_i = i * NxN;
        for (j = 0; j < N; j++){
            m[ind_i + j*N + 0] = 20.0;
            m[ind_i + j*N + Nm1] = 20.0;
        }
    }
}
/*
void
boundary(double *m, int N)
{
    int i, j, k;
    int ind_i, ind_j, ind_j2;
    int Nm1 = N - 1;
    int NxN = N * N;

    for (i = 0; i < N; i++){
        ind_i = i*NxN;
        for (k = 0; k < N; k++){
            m[ind_i + k] = 0.0;
        }
    }

    for (i = 0; i < N; i++){
        ind_i = i*NxN + Nm1*N;
        for (k = 0; k < N; k++){
            m[ind_i + k] = 20.0;
        }
    }

    for (j = 0; j < N; j++){
        ind_j = j * NxN;
        ind_j2 = Nm1*NxN + ind_j;
        for (k = 0; k < N; k++){
            m[ind_j + k] = 20.0;
            m[ind_j2 + k] = 20.0;
        }
    }

    for (i = 0; i < N; i++){
        ind_i = i * NxN;
        for (j = 0; j < N; j++){
            m[ind_i + j*N + 0] = 20.0;
            m[ind_i + j*N + Nm1] = 20.0;
        }
    }
}
*/

void
source(double *m, int N)
{
    int i, j, k;
    int ind_i, ind_j;
    int xl, xu, yl, yu, zl, zu;
    int NxN = N * N;

    double gs;

    xl = 0;
    xu = 5 * N / 16;
    yl = 0;
    yu = N / 4;
    zl = N / 6;
    zu = N / 2;

    gs = 200.0 * (2.0 / (N - 1.0)) * (2.0 / (N - 1.0));

    for (i = xl; i < xu; i++){
        ind_i = i*NxN;
        for (j = yl; j < yu; j++){
            ind_j = j * N;
            for (k = zl; k < zu; k++){
                m[ind_i + ind_j + k] = gs;
            }
        }
    }
}

void
initialize(double *u1, double *u2, double *f, int N, double start_T)
{
    full(u1, N, start_T);
    full(u2, N, start_T);
    zeros(f, N);

    boundary(u1, N);
    boundary(u2, N);
    source(f, N);
}