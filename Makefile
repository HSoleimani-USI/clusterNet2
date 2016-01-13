ROOT_DIR_CCP := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/cpp_source
ROOT_DIR_CU := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/cuda_source
ROOT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/build
NERVANA_DIR := /home/tim/git/nervanagpu/nervanagpu/kernels/C_interface
LEVELDB_DIR := /home/tim/git/leveldb
INCLUDE := -I /usr/local/cuda/include -I /home/tim/git/json/src/ -I $(ROOT_DIR)/include -I $(NERVANA_DIR) -I $(LEVELDB_DIR)/include
LIB := -L /usr/local/cuda/lib64 -L $(NERVANA_DIR) -L $(LEVELDB_DIR) -lnervana -lcudart -lcuda -lcublas -lcurand -lleveldb -lboost_system -lboost_date_time -lboost_thread -lboost_system  
FILES := $(ROOT_DIR_CU)/basicOps.cu $(ROOT_DIR_CU)/clusterKernels.cu $(ROOT_DIR_CU)/Timer.cu 
FILES_CPP := $(ROOT_DIR_CCP)/clusterNet2.cpp $(ROOT_DIR_CCP)/pythonWrapper.c $(ROOT_DIR_CCP)/pythonInterface.c $(ROOT_DIR_CCP)/BatchAllocator.cpp $(ROOT_DIR_CCP)/Layer.cpp $(ROOT_DIR_CCP)/NeuralNetwork.cpp
COMPUTE_CAPABILITY := arch=compute_52,code=sm_52 

all:	
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(ROOT_DIR)/build
	nvcc -gencode $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/Timer.o -o $(BUILD_DIR)/link.o 
	g++ -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/Timer.o  $(BUILD_DIR)/link.o $(FILES_CPP) -o $(ROOT_DIR)/lib/libclusternet2.so $(LIB) 	
	g++ -std=c++11 $(INCLUDE) -L $(ROOT_DIR)/lib $(ROOT_DIR)/main.cpp -o main.o $(LIB) -lclusternet2  
clean:
	rm *.o *.out $(ROOT_DIR)/py_source/gpupylib.so
