HDF5_DIR = /home/tim/apps/hdf5-1.8.14/hdf5
ROOT_DIR_CCP := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/cpp_source
ROOT_DIR_CU := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/cuda_source
ROOT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/build
NERVANA_DIR := /home/tim/git/nervanagpu/nervanagpu/kernels/C_interface
LEVELDB_DIR := /home/tim/git/leveldb
INCLUDE := -I /usr/local/cuda/include -I /home/tim/git/json/src/ -I $(ROOT_DIR)/include -I $(NERVANA_DIR) -I $(LEVELDB_DIR)/include
LIB := -L /usr/local/cuda/lib64 -L $(HDF5_DIR)lib -L $(NERVANA_DIR) -L $(LEVELDB_DIR) 
FLAGS := -lcudart -lcuda -lcublas -lcurand -lleveldb -lboost_system -lboost_date_time -lboost_thread -lboost_system -lhdf5 -lhdf5_hl
FILES := $(ROOT_DIR_CU)/BasicOpsCUDA.cu $(ROOT_DIR_CU)/clusterKernels.cu $(ROOT_DIR_CU)/Timer.cu $(ROOT_DIR_CCP)/Matrix.cpp 
FILES_CPP := $(wildcard $(ROOT_DIR_CCP)/*.cpp) $(wildcard $(ROOT_DIR_CCP)/*.c)
FILES_OUT := $(wildcard $(BUILD_DIR)/*.o) 
COMPUTE_CAPABILITY := arch=compute_52,code=sm_52 
SOURCES := $(shell find $(BUILD_DIR) -name '*.o')
all:	
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB) -lnervana $(FLAGS) --output-directory $(ROOT_DIR)/build 
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/Matrix.o $(BUILD_DIR)/BasicOpsCUDA.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/Timer.o -o $(BUILD_DIR)/link.o 
	g++ -std=c++11 -shared -fPIC $(INCLUDE) $(wildcard $(BUILD_DIR)/*.o) $(FILES_CPP) -o $(ROOT_DIR)/lib/libClusterNet.so $(LIB) -lnervana $(FLAGS) ++
kepler:
	nvcc -gencode arch=compute_30,code=sm_30 -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB) $(FLAGS) --output-directory $(ROOT_DIR)/build 
	nvcc -gencode arch=compute_30,code=sm_30 -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/Matrix.o $(BUILD_DIR)/BasicOpsCUDA.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/Timer.o -o $(BUILD_DIR)/link.o 
	g++ -std=c++11 -shared -fPIC $(INCLUDE) $(wildcard $(BUILD_DIR)/*.o) $(FILES_CPP) -o $(ROOT_DIR)/lib/libClusterNet.so $(LIB) $(FLAGS)	
test:
	g++ -std=c++11 $(INCLUDE) -L $(ROOT_DIR)/lib $(ROOT_DIR)/main.cpp -o main.o $(LIB) $(FLAGS) -lClusterNet  
clean:
	rm *.o *.out $(ROOT_DIR)/py_source/gpupylib.so
