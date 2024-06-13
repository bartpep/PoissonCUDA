# Variables
NVCC = nvc++
C = g++
CFLAGS = -g  -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=pinned -gpu=lineinfo -gpu=cc90 -cuda -mp=noautopar

TARGET_SEQ = 	jacobi_seq
TARGET_OPEN = 	jacobi_open
TARGET_CUDA = 	jacobi_cuda

SRCS_SEQ =  	main.cpp 		./jacobiSequential/jacobi.cpp 	$(wildcard ./auxiliaryFunctions/*.cpp) 
SRCS_OPEN = 	main.cpp 		./jacobiOpen/jacobi.cpp 		$(wildcard ./auxiliaryFunctions/*.cpp) 
SRCS_CUDA_CU = 	main_cuda.cu 	./jacobiCUDA/jacobi.cu 			$(wildcard ./auxiliaryFunctions/*.cu) 
SRCS_CUDA_CPP = 												$(wildcard ./auxiliaryFunctions/*.cpp) 


OBJS_SEQ = 	$(SRCS_SEQ:.cpp=.o)
OBJS_OPEN = $(SRCS_MP:.cpp=.o)
OBJS_CUDA = $(SRCS_CUDA_CPP:.cpp=.o) $(SRCS_CUDA_CU:.cu=.o) 


INCL_AUX = 	auxiliaryFunctions
INCL_S = 	jacobiSequential
INCL_OPEN = jacobiOpen
INCL_C = 	jacobiCUDA

all: jacobi_cuda jacobi_open
# Compile Jacobi cuda version
jacobi_cuda: $(TARGET_CUDA)
$(TARGET_CUDA): $(OBJS_CUDA)
	$(NVCC) $(CFLAGS) -o $@ $(OBJS_CUDA)  -I $(INCL_AUX) -I $(INCL_C)

jacobi_open: $(TARGET_OPEN)
$(TARGET_OPEN): $(OBJS_OPEN)
	$(NVCC) $(CFLAGS) -o $@ $(OBJS_OPEN)  -I $(INCL_AUX) -I $(INCL_OPEN)

jacobi_seq: $(TARGET_SEQ)
$(TARGET_SEQ): $(OBJS_SEQ)
	$(NVCC) $(CFLAGS) -o $@ $(OBJS_SEQ)  -I $(INCL_AUX) -I $(INCL_OPEN)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@ -I $(INCL_AUX) -I $(INCL_C)

%.o: %.cpp
	$(NVCC) $(CFLAGS) -c $< -o $@ -I $(INCL_AUX)


clean:
	rm -f jacobi_cuda *.o ./auxiliaryFunctions/*.o ./jacobiCUDA/*.o ./jacobiOpen/*.o

run_cuda:
	./jacobi_cuda 6 6 6
