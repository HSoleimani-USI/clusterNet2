#include <iostream>
#include <BasicOpsWrapper.h>

#ifdef NERVANA
	#include "nervana_c_api.h"
#endif


#ifndef __ClusterNet_H__
#define __ClusterNet_H__

typedef enum TransformerType_t
{
	DropoutTransform = 0
} TransformerType_t;

typedef enum WeightInitType_t
{
	Gaussian = 0,
	UniformSqrt = 1
} WeightInitType_t;

typedef enum Unittype_t
{
	Logistic = 0,
	Rectified_Linear = 1,
	Softmax = 2,
	Linear = 3,
	Input = 4,
	Exponential_linear = 5,
	Output

} Unittype_t;

typedef enum DataPropagationType_t
{
	Training = 0,
	Trainerror = 1,
	CVerror = 2
} DataPropagationType_t;


typedef enum Costfunction_t
{
	Cross_Entropy = 0,
	Squared_Error = 1,
	Root_Squared_Error = 2,
	Misclassification = 4
} Costfunction_t;

typedef enum Layer_t
{
	FullyConnected = 0,
	LSTM = 1,
	Lookup = 2,
	InputLayer = 3,
	OutputLayer = 4
} Layer_t;



class ClusterNet
{
    public:
        ClusterNet(){};
        ~ClusterNet(){};
		bool useNervanaGPU;

  		virtual void setRandomState(int seed) = 0;
  		virtual Matrix<float> *rand(int rows, int cols) = 0;
  		virtual Matrix<float> *randn(int rows, int cols) = 0;
  		virtual Matrix<float> *normal(int rows, int cols, float mean, float std) = 0;

        //  initializing the weight of network
  		virtual Matrix<float> *get_uniformsqrt_weight(int input, int output) = 0;

  		//  matrix multliplication A*B = out
  		virtual void Tdot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
  		virtual void dotT(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
  		// nothing transpose
  		virtual void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
  		//  define if the first and second are transpose
  		virtual void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2) = 0;
  		//  for regularization , using random numbers
  		//  getting the simplest model ...random between 0 and 1 , matrix
  		virtual void dropout(Matrix<float> *A, Matrix <float> *out, const float dropout) = 0;

  		BasicOpsWrapper *OPS;


};



#endif //__ClusterNet_H__
