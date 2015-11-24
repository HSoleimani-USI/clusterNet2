template<typename T> __global__ void kFill_with(T *m, T fill_value, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       m[i] = fill_value;
}

template __global__ void kFill_with<float>(float *m, float fill_value, int size);
template __global__ void kFill_with<int>(int *m, int fill_value, int size);


