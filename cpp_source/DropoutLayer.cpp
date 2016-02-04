#include <DropoutLayer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

DropoutLayer::DropoutLayer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu){ init(unitcount, start_batch_size,unit,gpu); }
DropoutLayer::DropoutLayer(int unitcount, Unittype_t unit){ init(unitcount, 0,unit, NULL); }
DropoutLayer::DropoutLayer(int unitcount){ init(unitcount, 0,Rectified_Linear, NULL); }


DropoutLayer::DropoutLayer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu)
{ init(unitcount, start_batch_size,unit,gpu); prev->link_with_next_Layer(this); }
DropoutLayer::DropoutLayer(int unitcount, Unittype_t unit, Layer *prev){ init(unitcount, 0,unit, prev->GPU); prev->link_with_next_Layer(this); }
DropoutLayer::DropoutLayer(int unitcount, Layer *prev){ init(unitcount, 0,Rectified_Linear, NULL); prev->link_with_next_Layer(this); }



void DropoutLayer::link_with_next_Layer(Layer *next_FCLayer)
{

	next = next_FCLayer;
	next->Layer_ID = Layer_ID + 1;
	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->GPU){next->GPU = GPU;}


	next->out = zeros<float>(BATCH_SIZE, next->UNITCOUNT);
	next->activation = zeros<float>(BATCH_SIZE, next->UNITCOUNT);
	next->error = zeros<float>(BATCH_SIZE, next->UNITCOUNT);

	next->bias_activations = ones<float>(1, BATCH_SIZE);
	next->prev = this;


	w_next = next->w_next;
	error = next->w_next;
}




void DropoutLayer::unit_activation(){ unit_activation(true); }
void DropoutLayer::unit_activation(bool useDropout)
{
	switch(UNIT_TYPE)
	{
		case Logistic:
			elementWiseUnary<klogistic>(out, activation, 0.0f);
			break;
		case Rectified_Linear:
			elementWiseUnary<krectified>(out, activation, 0.0f);
			break;
		case Softmax:
			softmax(out,out);
			break;
		case Linear:
			elementWiseUnary<kcopy>(out, activation,0.0f);
			break;
		case Input:
			break;
	}


	if(UNIT_TYPE != Softmax)
	{
		if(!useDropout)
			elementWiseUnary<ksmul>(activation,out,(1.0f-DROPOUT));
	}



}

void DropoutLayer::apply_dropout()
{
	if(UNIT_TYPE != Softmax)
	{
		GPU->dropout(activation,out,DROPOUT);
	}
}

void DropoutLayer::activation_gradient()
{

	switch(UNIT_TYPE)
	{
		case Logistic:
			elementWiseUnary<klogistic>(activation, out, 0.0f);
			break;
		case Rectified_Linear:
			elementWiseUnary<krectified_grad>(activation, out, 0.0f);
			break;
		case Softmax:
			break;
		default:
			throw "Unknown unit";
			break;
	}

}

void DropoutLayer::forward(){ forward(true); }
void DropoutLayer::forward(bool useDropout)
{

	if(_network->_isTrainTime)
	{
		GPU->dropout(prev->activation,activation,DROPOUT);
	}
	else
	{
		elementWiseUnary<ksmul>(prev->activation,activation,(1.0f-DROPOUT));
	}
    if(next){ next->forward(useDropout); }
}


void DropoutLayer::backward_errors()
{
	if(!target){ next->backward_errors(); }
	if(target)
	{
		if(out->cols != target->cols && !target_matrix){ target_matrix = zeros<float>(BATCH_SIZE,out->cols); }
		if(out->cols != target->cols)
		{
			vectorWise<ktmatrix>(target,target, target_matrix,0.0f);
			elementWise<ksub>(out,target_matrix,error,0.0f); return;
		}
		else{ elementWise<ksub>(activation,target,error,0.0f);  return;}


		elementWiseUnary<ksmul>(out,out,1.0f/error->rows); return;
	}

	if(UNIT_TYPE == Input){ backward_grads(); return; }


	activation_gradient();
	GPU->dotT(next->error, w_next,error);
	elementWise<kmul>(error, out, error,0.0f);

}

void DropoutLayer::backward_grads()
{
	GPU->Tdot(activation, next->error, w_grad_next);
	if(!next->target){ next->backward_grads(); }
	GPU->dot(next->bias_activations, next->error,b_grad_next);
}





