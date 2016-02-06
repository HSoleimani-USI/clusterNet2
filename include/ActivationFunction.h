#ifndef ActivationFunction_H
#define ActivationFunction_H

#include <ClusterNet.h>

template <typename T> class Matrix;

class ActivationFunction
{

public:
	ActivationFunction(Unittype_t unitType);
	~ActivationFunction(){};
	void activation(Matrix<float> *in, Matrix<float> *out);
	void activation_gradient(Matrix<float> *in, Matrix<float> *out);
	Unittype_t _unitType;
private:

};

#endif
