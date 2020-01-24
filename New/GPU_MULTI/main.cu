/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "print.h"
#include "init.h"
#include "jacobi.h"

#define N_DEFAULT 100

int
main(int argc, char *argv[]) 
{

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance=1000;
    double	start_T=0;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char    *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	*h_u1 = NULL;
    double 	*h_u2 = NULL;
    double 	*h_f = NULL;


    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }
    
    long long total_size = N * N * N * sizeof(double);


    // allocate memory
    cudaMallocHost((void**)&h_u1, total_size);
    cudaMallocHost((void**)&h_u2, total_size);
    cudaMallocHost((void**)&h_f, total_size);




    // check allocation
    if ( h_u1 == NULL ) {
        perror("array h_u1: allocation failed");
        exit(-1);
    }
    if ( h_u2 == NULL ) {
        perror("array h_u2: allocation failed");
        exit(-1);
    }
    if ( h_f == NULL ) {
        perror("array h_f: allocation failed");
        exit(-1);
    }
    // if ( d_u1 == NULL ) {
    //     perror("array d_u1: allocation failed");
    //     exit(-1);
    // }
    // if ( d_u2 == NULL ) {
    //     perror("array d_u2: allocation failed");
    //     exit(-1);
    // }
    // if ( d_f == NULL ) {
    //     perror("array d_f: allocation failed");
    //     exit(-1);
    // }

    // initialize arrays on host
    initialize(h_u1, h_u2, h_f, N, start_T);


    // calculate poisson problem with the jacobi method
    jacobi(h_u1, h_u2, h_f, N, iter_max, tolerance);


    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, h_u1);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N, h_u1);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }
    

    // de-allocate memory
    cudaFree(h_u1);
    cudaFree(h_u2);
    cudaFree(h_f);


    return(0);
}
