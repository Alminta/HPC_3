TARGET	= poisson_naive

# /*
# SOURCES	= main.c print.c alloc3d.c init.c
# OBJECTS	= print.o alloc3d.o init.o
# MAIN	= main.o
# OBJS	= $(MAIN) jacobi.o
# */

OBJS	= main.o jacobi.o print.o init.o #alloc.o

OPT	    = -O3
PARA	= -fopenmp
XOPTS   = -Xptxas=-v -arch=sm_70 -lineinfo

CC	= gcc
CCC	= nvcc
CXX	= nvcc

CFLAGS	= $(OPT) $(PARA) $(XOPT)
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

XLIBS	= -lcublas
SOFLAGS = -shared

CUDA_PATH ?= /appl/cuda/10.2
INCLUDES = -I$(CUDA_PATH)/include \
	       -I$(CUDA_PATH)/samples/NVIDIA_CUDA-10.2_Samples/common/inc

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(XLIBS)

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET)

# dependencies

# alloc.o: alloc.cu alloc.h
init.o: init.cu init.h
jacobi.o: jacobi.cu jacobi.h
main.o: main.cu print.h jacobi.h init.h
print.o: print.cu print.h

