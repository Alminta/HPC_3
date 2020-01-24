#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q hpcintrogpu
### -- set the job Name --
#BSUB -J gpu[1-4]
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10
# request 5GB of system-memory
#BSUB -R "rusage[mem=2GB]"
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu%I-%J.out
# -- end of LSF options --

set -x

nvidia-smi
# Load the cuda module
module load cuda/10.2
module load gcc/8.2.0

/appl/cuda/9.1/samples/bin/x86_64/linux/release/deviceQuery

export MATMULT_COMPARE=0
export MFLOPS_MAX_IT=50

N=32
N2=64
N3=128
N4=256
N5=501
N6=1024
N7=2048
N8=4096
N9=8192
N10=16384

SIZE="$N $N $N"
SIZE2="$N2 $N2 $N2"
SIZE3="$N3 $N3 $N3"
SIZE4="$N4 $N4 $N4"
SIZE5="$N5 $N5 $N5"
SIZE6="$N6 $N6 $N6"
SIZE7="$N7 $N7 $N7"
SIZE8="$N8 $N8 $N8"
SIZE9="$N9 $N9 $N9"
SIZE10="$N10 $N10 $N10"

./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE2
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE3
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE4
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE5
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE6
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE7
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE8
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE9
./matmult_f.nvcc gpu$LSB_JOBINDEX $SIZE10

# ./matmult_f.nvcc gpulib $SIZE
# ./matmult_f.nvcc gpulib $SIZE2
# ./matmult_f.nvcc gpulib $SIZE3
# ./matmult_f.nvcc gpulib $SIZE4
# ./matmult_f.nvcc gpulib $SIZE5
# ./matmult_f.nvcc gpulib $SIZE6
# ./matmult_f.nvcc gpulib $SIZE7
# ./matmult_f.nvcc gpulib $SIZE8
# ./matmult_f.nvcc gpulib $SIZE9
# ./matmult_f.nvcc gpulib $SIZE10

# export OMP_NUM_THREADS=12
# export OMP PROC_BIND=close
# ./matmult_f.nvcc lib $SIZE
# ./matmult_f.nvcc lib $SIZE2
# ./matmult_f.nvcc lib $SIZE3
# ./matmult_f.nvcc lib $SIZE4
# ./matmult_f.nvcc lib $SIZE5
# ./matmult_f.nvcc lib $SIZE6
# ./matmult_f.nvcc lib $SIZE7
# ./matmult_f.nvcc lib $SIZE8
# ./matmult_f.nvcc lib $SIZE9
# ./matmult_f.nvcc lib $SIZE10