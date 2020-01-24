#!/bin/bash
#BSUB -J xbone_1_seq
#BSUB -o xbone_1_seq_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 10
#BSUB -R "span[hosts=1]"

# set -x

nvidia-smi

# module load studio
# module load clang/9.0.0
# module swap clang/9.0.0
module load cuda/10.2
module load gcc/8.3.0
# /appl/cuda/9.1/samples/bin/x86_64/linux/release/deviceQuery

# executable
EXECUTABLE=poisson_seq

# args
N="64 128 256"
ITER="100"
TOL="0"
START_T="0"
OUT="0"

# environment variables

NUM_RUNS=10
ITERS="10 100 1000"
# start the collect command with the above settings
# ./$EXECUTABLE $N $ITER $TOL $START_T $OUT

# time for i in {1..1000}; do ./$EXECUTABLE 25 $ITER $TOL $START_T $OUT; done
# time for i in {1..100}; do ./$EXECUTABLE 50 $ITER $TOL $START_T $OUT; done
# time for i in {1..10}; do ./$EXECUTABLE 100 $ITER $TOL $START_T $OUT; done
# time ./$EXECUTABLE 200 $ITER $TOL $START_T $OUT

for n in $N
do
    for i in $ITERS
    do
        time ./$EXECUTABLE $n $i $TOL $START_T $OUT
        nvprof --print-gpu-summary ./$EXECUTABLE $n $i $TOL $START_T $OUT
    done
done


# for n in $N
# do
#     time ./$EXECUTABLE $n $ITER $TOL $START_T $OUT
#     nvprof --print-gpu-summary ./$EXECUTABLE $n $ITER $TOL $START_T $OUT
# done



# for n in $N
# do
#     for j in RUNS
#     do
#         time for i in {1..$j}; do ./$EXECUTABLE $n $ITER $TOL $START_T $OUT; done
#     done
# done

# for n in $N
# do
#     for T in $THREADS
#     do
#         time OMP_NUM_THREADS=$T ./$EXECUTABLE $n $ITER $TOL $START_T $OUT
#     done
# done

# for n in $N
# do
#     time ./$EXECUTABLE $n $ITER $TOL $START_T $OUT
# done

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