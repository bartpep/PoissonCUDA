# Variables
ITERATIONS = 10000
N = 250
DIFFERENCE = .001
THREAD_NUM = 4

# Make file
NVCC = nvc++
CFLAGS = -g  -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=pinned -gpu=lineinfo -gpu=cc90 -cuda -mp=noautopar

CC = g++
MPFLAGS = -g -O3 -fopenmp -fopt-info -ffast-math -funroll-loops


TARGET_SEQ = 	seq
TARGET_OPEN = 	open
TARGET_CUDA = 	cuda

SRCS_SEQ =  	main.cpp 		./jacobiSequential/jacobi.cpp 	$(wildcard ./auxiliaryFunctions/*.cpp) 
SRCS_OPEN = 	main.cpp 		./jacobiOpen/jacobi.cpp 		$(wildcard ./auxiliaryFunctions/*.cpp) 
SRCS_CUDA_CU = 	main_cuda.cu 	./jacobiCUDA/jacobi.cu 			$(wildcard ./auxiliaryFunctions/*.cu) 
SRCS_CUDA_CPP = 												$(wildcard ./auxiliaryFunctions/*.cpp) 


OBJS_SEQ = 	$(SRCS_SEQ:.cpp=.o)
OBJS_OPEN = $(SRCS_OPEN:.cpp=.open.o)
OBJS_CUDA = $(SRCS_CUDA_CPP:.cpp=.cuda.o) $(SRCS_CUDA_CU:cu=.cuda.o) 


INCL_AUX = 	auxiliaryFunctions
INCL_SEQ = 	jacobiSequential
INCL_OPEN = jacobiOpen
INCL_C = 	jacobiCUDA

# Compile Jacobi cuda version
cuda: $(TARGET_CUDA)
$(TARGET_CUDA): $(OBJS_CUDA)
	$(NVCC) $(CFLAGS) -o $@ $(OBJS_CUDA)  -I $(INCL_AUX) -I $(INCL_C)

%cuda.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@ -I $(INCL_AUX) -I $(INCL_C) 

%cuda.o: %.cpp
	$(NVCC) $(CFLAGS) -c $< -o $@ -I $(INCL_AUX) -I $(INCL_SEQ) -I $(INCL_OPEN)

open: $(TARGET_OPEN)
$(TARGET_OPEN): $(OBJS_OPEN)
	$(CC) $(MPFLAGS) -o $@ $(OBJS_OPEN)  -I $(INCL_AUX) -I $(INCL_OPEN)

%.open.o: %.cpp
	$(CC) $(MPFLAGS) -c $< -o $@ -I $(INCL_AUX) -I $(INCL_SEQ) -I $(INCL_OPEN)

seq: $(TARGET_SEQ)
$(TARGET_SEQ): $(OBJS_SEQ)
	$(NVCC) $(CFLAGS) -o $@ $(OBJS_SEQ)  -I $(INCL_AUX) -I $(INCL_SEQ)

%.o: %.cu
	$(CC)  -c $< -o $@ -I $(INCL_AUX) -I $(INCL_C) 

%.o: %.cpp
	$(CC)  -c $< -o $@ -I $(INCL_AUX) -I $(INCL_SEQ) -I $(INCL_OPEN)


clean:
	rm -f jacobi_cuda *.o ./auxiliaryFunctions/*.o ./jacobiCUDA/*.o ./jacobiOpen/*.o ./jacobiSequential/*.o
	rm -f jacobi_cuda jacobi_open jacobi_seq

run_seq:
	./seq $(N) $(ITERATIONS) $(DIFFERENCE) SEQ

run_open:
	OMP_NUM_THREADS=$(THREAD_NUM) ./open $(N) $(ITERATIONS) $(DIFFERENCE) MP

run_cuda:
	./cuda $(N) $(N) $(N) $(ITERATIONS) $(DIFFERENCE) 



