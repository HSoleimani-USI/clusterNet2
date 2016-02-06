#include <ActivationFunction.h>
#include <basicOps.cuh>


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
				elementWiseUnary<klogistic>(in, out, 0.0f);
				break;
			case Rectified_Linear:
				elementWiseUnary<krectified>(in, out, 0.0f);
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
			elementWiseUnary<klogistic>(in, out, 0.0f);
			break;
		case Rectified_Linear:
			elementWiseUnary<krectified_grad>(in, out, 0.0f);
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

