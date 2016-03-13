#include <ActivationFunction.h>
#include <BasicOpsWrapper.h>


using std::cout;
using std::endl;

ActivationFunction::ActivationFunction(Unittype_t unitType, ClusterNet *gpu)
{
	_unitType = unitType;
	GPU = gpu;
}

void ActivationFunction::activation(Matrix<float> *in, Matrix<float> *out)
{
	switch(_unitType)
		{
			case Logistic:
				GPU->OPS->logistic(in, out);
				break;
			case Rectified_Linear:
				GPU->OPS->rectified(in, out);
				break;
			case Exponential_linear:
				GPU->OPS->ELU(in, out);
				break;
			case Softmax:
				GPU->OPS->softmax(in, out);
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
			GPU->OPS->logistic_grad(in, out);
			break;
		case Rectified_Linear:
			GPU->OPS->rectified_grad(in, out);
			break;
		case Exponential_linear:
			GPU->OPS->ELU_grad(in, out);
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

