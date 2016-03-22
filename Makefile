HDF5_DIR = /home/tim/apps/hdf5-1.8.14/hdf5
ROOT_DIR_CCP := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/cpp_source
ROOT_DIR_CU := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/cuda_source
ROOT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/build
NERVANA_DIR := /home/tim/git/nervanagpu/nervanagpu/kernels/C_interface
INCLUDE :=  -I /usr/local/cuda/include -I $(ROOT_DIR)/include -I $(NERVANA_DIR) -I $(HDF5_INC_DIR)
LIB := -L /usr/local/cuda/lib64 -L $(HDF5_DIR)lib -L $(NERVANA_DIR) -L $(HDF5_LIB_DIR)
FLAGS_GPU := -lcudart -lcuda -lcublas -lcurand -lhdf5 -lhdf5_hl -lcblas
FLAGS_CPU := -lhdf5 -lhdf5_hl -lcblas
FLAGS_PHI := -lhdf5 -lhdf5_hl -mkl -openmp
FILES := $(ROOT_DIR_CU)/BasicOpsCUDA.cu $(ROOT_DIR_CU)/clusterKernels.cu $(ROOT_DIR_CU)/Timer.cu $(ROOT_DIR_CCP)/Matrix.cpp 
FILES_CPP := $(wildcard $(ROOT_DIR_CCP)/*.cpp) $(wildcard $(ROOT_DIR_CCP)/*.c)
FILES_CPP_GPU := $(wildcard $(ROOT_DIR_CCP)/GPU/*.cpp) $(wildcard $(ROOT_DIR_CCP)/GPU/*.c)
FILES_OUT := $(wildcard $(BUILD_DIR)/*.o) 
COMPUTE_CAPABILITY := arch=compute_52,code=sm_52 
SOURCES := $(shell find $(BUILD_DIR) -name '*.o')
all:
	
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB) -lnervana $(FLAGS_GPU) --output-directory $(ROOT_DIR)/build 
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/Matrix.o $(BUILD_DIR)/BasicOpsCUDA.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/Timer.o -o $(BUILD_DIR)/link.o
	g++ -D NERVANA -std=c++11 -shared -fPIC $(INCLUDE)  $(wildcard $(BUILD_DIR)/*.o) $(FILES_CPP) $(FILES_CPP_GPU) -o $(ROOT_DIR)/lib/libClusterNet.so $(LIB) -lnervana $(FLAGS_GPU)
kepler:
	nvcc -gencode arch=compute_30,code=sm_30 -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB) $(FLAGS) $(FLAGS_GPU) --output-directory $(ROOT_DIR)/build 
	nvcc -gencode arch=compute_30,code=sm_30 -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/Matrix.o $(BUILD_DIR)/BasicOpsCUDA.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/Timer.o -o $(BUILD_DIR)/link.o 
	g++ -std=c++11 -shared -fPIC $(INCLUDE) $(wildcard $(BUILD_DIR)/*.o) $(FILES_CPP) $(FILES_CPP_GPU) -o $(ROOT_DIR)/lib/libClusterNet.so $(LIB) $(FLAGS_GPU)	
c:
	g++ -std=c++11 -shared -fPIC $(INCLUDE) $(FILES_CPP) -o $(ROOT_DIR)/lib/libClusterNet.so $(LIB) $(FLAGS_CPU)	
test:
	g++ -std=c++11 $(INCLUDE) -L $(ROOT_DIR)/lib $(ROOT_DIR)/main.cpp -o main.o $(LIB) $(FLAGS_GPU) -lClusterNet  
testc:
	g++ -std=c++11 $(INCLUDE) -L $(ROOT_DIR)/lib $(ROOT_DIR)/mainCPU.cpp -o main.o -L $(HDF5_DIR)lib $(FLAGS_CPU) -lClusterNet
	
phi:
	icpc -D PHI -std=c++11 -shared -fPIC $(INCLUDE) $(FILES_CPP) -o $(ROOT_DIR)/lib/libClusterNet.so $(LIB) $(FLAGS_PHI)	
testphi:
	icpc -D PHI -std=c++11 $(INCLUDE) -L $(ROOT_DIR)/lib $(ROOT_DIR)/mainPhi.cpp -o main.o -L $(HDF5_LIB_DIR) $(FLAGS_PHI) -lClusterNet

	
clean:
	rm $(ROOT_DIR)/lib/libClusterNet.so
	rm build/*.o 
