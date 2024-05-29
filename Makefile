# Variables
C = g++
NVCC = nvcc
CFLAGS_MP = -ftree-parallelize-loops=4
NVCC_FLAGS = 
TARGET_SEQ = 	jacobi_seq
TARGET_MP = 	jacobi_open
TARGET_CUDA = 	jacobi_cuda

SRCS_SEQ =  main.cpp 	./jacobiSequential/jacobi.cpp 	./auxiliaryFunctions/print.cpp ./auxiliaryFunctions/initial_functions.cpp
SRCS_MP = 	main.cpp 	./jacobiOpen/jacobi.cpp 		./auxiliaryFunctions/print.cpp ./auxiliaryFunctions/initial_functions.cpp
SRCS_CUDA = main.cpp 	./jacobiCUDA/jacobi.cu 			./auxiliaryFunctions/print.cpp ./auxiliaryFunctions/initial_functions.cpp

OBJS_SEQ = 	$(SRCS_SEQ:.cpp=.o)
OBJS_MP = 	$(SRCS_MP:.cpp=.o)
OBJS_CUDA = $(SRCS_CUDA:.cpp=.o)
OBJS_CUDA = $(SRCS_CUDA:.cpp=.o)

INCL_AUX = 	auxiliaryFunctions
INCL_S = 	jacobiSequential
INCL_MP = 	jacobiOpen
INCL_C = 	jacobiCUDA

# Targets
all: $(TARGET_SEQ)

# Compile Sequential version
seq: $(TARGET_SEQ)
$(TARGET_SEQ): $(OBJS_SEQ)
	$(C) $(CFLAGS) -o $(TARGET_SEQ) $(OBJS_SEQ) -I $(INCL_AUX) -I $(INCL_S) -fopenmp

%.o: %.cpp
	$(C) $(CFLAGS) -c $< -o $@ 



clean_seq:
	rm -f $(OBJS_SEQ) $(TARGET_SEQ)

# Compile Jacobi OpenMP version
open:  $(TARGET_MP)
$(TARGET_MP): $(OBJS_MP)
	$(C) $(CFLAGS_MP) -o $(TARGET_MP) $(OBJS_MP) -I $(INCL_AUX) -I $(INCL_MP) -MP

%.o: %.cpp
	$(C) $(CFLAGS) -c $< -o $@ 

clean_open:
	rm -f $(OBJS_MP) $(TARGET_MP)

# Compile Jacobi cuda version
cuda:  $(TARGET_CUDA)
$(TARGET_CUDA): $(OBJS_CUDA)
	$(NVCC) $(CFLAGS) -o $(TARGET_CUDA) $(OBJS_CUDA) -I $(INCL_AUX) -I $(INCL_CUDA) CUDA
	
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

%.o: %.cpp
	$(C) $(CFLAGS) -c $< -o $@ 



clean_cuda:
	rm -f $(OBJS_CUDA) $(TARGET_CUDA)




