#include <FCLayer.h>
#include <Network.h>
#include <ActivationFunction.h>
#include <Transformer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

FCLayer::FCLayer(int unitcount, Unittype_t unitType){ init(unitcount, unitType, FullyConnected); }

void FCLayer::forward()
{
	if(!prev){ apply_transformations(); next->forward(); return; }

	GPU->dot(prev->get_forward_activation(),prev->w_next,activation);
	GPU->OPS->vadd(activation, prev->b_next, activation);
	_func->activation(activation,activation);

	apply_transformations();

    if(next){ next->forward(); }
}


void FCLayer::backward_errors()
{
	if(!target){ next->backward_errors(); }

	if(target)
	{
		if(activation->cols != target->cols && !target_matrix){ target_matrix = GPU->OPS->zeros(BATCH_SIZE,activation->cols); }
		if(activation->cols != target->cols)
		{
			//cout << "activations: " << reduceToValue<rsum>(activation) << endl;
			GPU->OPS->get_t_matrix(target, target_matrix);
			GPU->OPS->sub(activation,target_matrix,error);
		}
		else{ GPU->OPS->sub(activation,target,error); }


		GPU->OPS->mul(error,error,1.0f/error->rows);
		//GPU->Tdot(prev->activation, error, prev->w_grad_next);

		return;

	}

	if(_func->_unitType == Input){ backward_grads(); return; }

	_func->activation_gradient(activation, activation_grad);
	GPU->dotT(next->error, w_next,error);
	GPU->OPS->mul(error, activation_grad, error);
	//GPU->Tdot(prev->activation, error, prev->w_grad_next);
}

void FCLayer::backward_grads()
{
	GPU->Tdot(activation, next->error, w_grad_next);
	if(!next->target){ next->backward_grads(); }
	GPU->OPS->reduceToColsMean(next->error,b_grad_next);
}


