#include <pythonWrapper.h>

FloatMatrix *fill_matrix(int rows, int cols, float fill_value){	return fill_matrix<float>(rows, cols, fill_value); }
FloatMatrix *empty(int rows, int cols){	return empty<float>(rows, cols); }
void to_host(FloatMatrix *gpu, float *cpu){ to_host<float>(gpu, cpu); }
void to_gpu(float *cpu, FloatMatrix *gpu){ to_gpu<float>(cpu, gpu); }
ClusterNet *get_clusterNet(){ return new ClusterNet2<float>(); }
