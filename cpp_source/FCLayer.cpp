#include <FCLayer.h>
#include <Network.h>
#include <ActivationFunction.h>
#include <Transformer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

FCLayer::FCLayer(int unitcount, Unittype_t unitType){ init(unitcount, unitType); }

void FCLayer::forward()
{
	if(!prev){ if(_transformer){ _transformer->transform(activation); } next->forward(); return; }

	GPU->dot(prev->get_forward_activation(),prev->w_next,activation);
	vectorWise<kvadd>(activation, prev->b_next, activation);
	_func->activation(activation,activation);

	if(_transformer){ _transformer->transform(activation); }

    if(next){ next->forward(); }
}


void FCLayer::backward_errors()
{
	if(!target){ next->backward_errors(); }

	if(target)
	{
		if(activation->cols != target->cols && !target_matrix){ target_matrix = zeros<float>(BATCH_SIZE,activation->cols); }
		if(activation->cols != target->cols)
		{
			//cout << "activations: " << reduceToValue<rsum>(activation) << endl;
			vectorWise<ktmatrix>(target, target_matrix);
			elementWise<ksub>(activation,target_matrix,error);
		}
		else{ elementWise<ksub>(activation,target,error); }


		elementWise<ksmul>(error,error,1.0f/error->rows);
		//GPU->Tdot(prev->activation, error, prev->w_grad_next);

		return;

	}

	if(_func->_unitType == Input){ backward_grads(); return; }

	_func->activation_gradient(activation, activation_grad);
	GPU->dotT(next->error, w_next,error);
	elementWise<kmul>(error, activation_grad, error);
	//GPU->Tdot(prev->activation, error, prev->w_grad_next);
}

void FCLayer::backward_grads()
{
	GPU->Tdot(activation, next->error, w_grad_next);
	if(!next->target){ next->backward_grads(); }
	reduceToCols<rmean>(next->error,b_grad_next);
}


