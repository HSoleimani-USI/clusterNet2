#include <ActivationFunction.h>
#include <BasicOpsCUDA.cuh>


using std::cout;
using std::endl;

ActivationFunction::ActivationFunction(Unittype_t unitType)
{
	_unitType = unitType;
}

void ActivationFunction::activation(Matrix<float> *in, Matrix<float> *out)
{
	switch(_unitType)
		{
			case Logistic:
				elementWise<klogistic>(in, out);
				break;
			case Rectified_Linear:
				elementWise<krectified>(in, out);
				break;
			case Exponential_linear:
				elementWise<kELU>(in, out);
				break;
			case Softmax:
				softmax(in,out);
				break;
			case Linear:
				break;
			case Input:
				break;
		}

}

void ActivationFunction::activation_gradient(Matrix<float> *in, Matrix<float> *out)
{
	switch(_unitType)
	{
		case Logistic:
			elementWise<klogistic>(in, out);
			break;
		case Rectified_Linear:
			elementWise<krectified_grad>(in, out);
			break;
		case Exponential_linear:
			elementWise<kELU_grad>(in, out);
			break;
		case Softmax:
			break;
		case Linear:
			break;
		default:
			cout << "Unknown unit" << endl;
			throw "Unknown unit";
			break;
	}
}

