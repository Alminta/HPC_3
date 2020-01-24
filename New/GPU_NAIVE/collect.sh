#!/bin/bash
#
#BSUB -J tyr_2
#BSUB -o tyr_2_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=1024]"
#BSUB -W 4
#BSUB -R "span[hosts=1]"
#BSUB -N

module load studio

# executable
EXECUTABLE=poisson_j

# args
N="10 20 30 40 50 60 70 80 90 100"
ITER="10000"
TOL="0.05"
START_T="0"
OUT="0"

# environment variables
export OMP_NUM_THREADS=1

# experiment name 
JID=${LSB_JOBID}
EXPOUT="$LSB_JOBNAME.${JID}.er"

# the example below is for, L1 misses, L2 misses, L3 misses
HWCOUNT="-h dcm,on,l2m,on,l3m,on"

# start the collect command with the above settings
collect -o $EXPOUT $HWCOUNT ./$EXECUTABLE $N $ITER $TOL $START_T $OUT

# counter=1
# while [ $counter -le 200 ]
# do
#     collect -o $EXPOUT $HWCOUNT ./$EXECUTABLE $PERM $MKN $counter
#     ((counter++))
# done

# array=( 1 2 4 8 16 32 64 128 256 )
# for i in "${array[@]}"
# do
#     collect -o $EXPOUT $HWCOUNT ./$EXECUTABLE $PERM $MKN $i
#     ((counter++))
# done