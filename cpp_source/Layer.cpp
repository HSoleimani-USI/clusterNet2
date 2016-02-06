#include <Layer.h>
#include <ActivationFunction.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

/*
void Layer::weight_update()
{
	if(target){ return; }

	next->weight_update();

	if(_LayerType == "Dropout"){ return; }

	switch(UPDATE_TYPE)
	{
		case RMSProp:
			RMSprop_with_weight_update(w_rms_next,w_grad_next,w_next,RMSPROP_MOMENTUM,LEARNING_RATE);
			RMSprop_with_weight_update(b_rms_next,b_grad_next,b_next,RMSPROP_MOMENTUM,LEARNING_RATE/100.0f);
			break;
		case PlainSGD:
			elementWiseUnary<ksmul>(w_grad_next,w_grad_next,LEARNING_RATE);
			elementWise<ksub>(w_next,w_grad_next,w_next,0.0f);
			break;
		default:
			cout << "Unknown update type!" << endl;
			throw "Unknown update type!";
			break;
	}

}

void Layer::print_error(string message)
{
	if(!target){ next->print_error(message); return;}

		cout << message << RUNNING_ERROR/RUNNING_SAMPLE_SIZE << endl;


	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
}

void Layer::set_hidden_dropout(float dropout)
{
	if(!next){ return; }
	next->DROPOUT = dropout;
	next->set_hidden_dropout(dropout);
}

void Layer::learning_rate_decay(float decay_rate)
{
	if(!next){ return; }
	next->LEARNING_RATE *= decay_rate;
	next->learning_rate_decay(decay_rate);
}

void Layer::dropout_decay()
{
	if(!prev){ cout << "Decaying dropout!" << endl; }
	if(!next){ return;}

	cout << "Setting dropout from " << DROPOUT << " to " << DROPOUT/2.0f << endl;
	DROPOUT /= 2.0f;
	next->dropout_decay();
}




void Layer::running_error()
{
	if(!target){ next->running_error(); return;}

	string text = "";

	if(!result)
	{
		result = empty<float>(target->rows,target->cols);
		eq = empty<float>(target->rows, target->cols);
	}

	float sum_value = 0.0f;

	switch(COST)
	{
		case Misclassification:
			argmax(activation, result);
			elementWise<keq>(result,target,eq,0.0f);
			sum_value = sum(eq);
			RUNNING_ERROR += (activation->rows  - sum_value);
			RUNNING_SAMPLE_SIZE += activation->rows;
			break;
		default:
			throw "Unknown cost function!";
			break;
	}
}


*/
void Layer::init(int unitcount, Unittype_t unitType)
{
	next = NULL;
	prev = NULL;
	w_next = NULL;
	b_next = NULL;
	w_rms_next = NULL;
	b_rms_next = NULL;
	b_grad_next = NULL;
	eq = NULL;
	result = NULL;

	target = NULL;
	target_matrix = NULL;
	error = NULL;

	UNITCOUNT = unitcount;

	Layer_ID = 0;

	_func = new ActivationFunction(unitType);
}

