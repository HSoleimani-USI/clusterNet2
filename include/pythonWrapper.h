#include <basicOps.cuh>

typedef Matrix<float> FloatMatrix;
FloatMatrix *fill_matrix(int rows, int cols, float fill_value);
FloatMatrix *empty(int rows, int cols);
