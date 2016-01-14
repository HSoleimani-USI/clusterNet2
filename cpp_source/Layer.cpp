#include <Layer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

Layer::Layer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<float> *gpu){ init(unitcount, start_batch_size,unit,gpu); }
Layer::Layer(int unitcount, Unittype_t unit){ init(unitcount, 0,unit, NULL); }
Layer::Layer(int unitcount){ init(unitcount, 0,Rectified_Linear, NULL); }


Layer::Layer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet2<float> *gpu)
{ init(unitcount, start_batch_size,unit,gpu); prev->link_with_next_layer(this); }
Layer::Layer(int unitcount, Unittype_t unit, Layer *prev){ init(unitcount, 0,unit, prev->GPU); prev->link_with_next_layer(this); }
Layer::Layer(int unitcount, Layer *prev){ init(unitcount, 0,Rectified_Linear, NULL); prev->link_with_next_layer(this); }


void Layer::init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<float> *gpu)
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

	LEARNING_RATE = 0.001f;
	RMSPROP_MOMENTUM = 0.9f;
	UNIT_TYPE = unit;
	DROPOUT = 0.5f;
	UNITCOUNT = unitcount;
	BATCH_SIZE = start_batch_size;
	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
	L2 = 15.0f;


	UPDATE_TYPE = RMSProp;
	//UPDATE_TYPE = NoMomentum;
	COST = Misclassification;
	LAYER_ID = 0;

	GPU = gpu;

	count = 0;


	if(BATCH_SIZE > 0)
	{
		out = zeros<float>(BATCH_SIZE, UNITCOUNT);
		bias_activations = ones<float>(1, BATCH_SIZE);
		activation = zeros<float>(BATCH_SIZE, UNITCOUNT);
	}
	else
	{
		out = NULL;
		bias_activations = NULL;
		activation = NULL;
	}

}


void Layer::link_with_next_layer(Layer *next_layer)
{

	next = next_layer;
	next->LAYER_ID = LAYER_ID + 1;
	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->GPU){next->GPU = GPU;}


	w_next = GPU->normal(UNITCOUNT,next_layer->UNITCOUNT,0.0f,0.01f);;
	w_rms_next = zeros<float>(UNITCOUNT,next_layer->UNITCOUNT);

	w_grad_next = zeros<float>(UNITCOUNT,next_layer->UNITCOUNT);

	b_next = zeros<float>(1,next_layer->UNITCOUNT);
	b_grad_next = zeros<float>(1,next_layer->UNITCOUNT);
	b_rms_next = zeros<float>(1,next_layer->UNITCOUNT);

	next->out = zeros<float>(BATCH_SIZE, next->UNITCOUNT);
	next->activation = zeros<float>(BATCH_SIZE, next->UNITCOUNT);
	next->error = zeros<float>(BATCH_SIZE, next->UNITCOUNT);

	next->bias_activations = ones<float>(1, BATCH_SIZE);
	next->prev = this;
}




void Layer::unit_activation(){ unit_activation(true); }
void Layer::unit_activation(bool useDropout)
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

void Layer::apply_dropout()
{
	if(UNIT_TYPE != Softmax)
	{
		GPU->dropout(activation,out,DROPOUT);
	}
}

void Layer::activation_gradient()
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

void Layer::forward(){ forward(true); }
void Layer::forward(bool useDropout)
{
	if(!prev){  unit_activation(useDropout); if(useDropout){apply_dropout(); } next->forward(useDropout); return; }



	GPU->dot(prev->out,prev->w_next,out);


	vectorWise<kvadd>(out, prev->b_next, out, 0.0f);
    unit_activation(useDropout);

    if(useDropout){apply_dropout(); }
    if(next){ next->forward(useDropout); }
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
			argmax(out, result);
			elementWise<keq>(result,target,eq,0.0f);
			sum_value = reduceToValue<rsum>(eq);
			RUNNING_ERROR += (out->rows  - sum_value);
			RUNNING_SAMPLE_SIZE += out->rows;
			break;
		default:
			throw "Unknown cost function!";
			break;
	}
}



void Layer::backward_errors()
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

void Layer::backward_grads()
{
	GPU->Tdot(activation, next->error, w_grad_next);
	if(!next->target){ next->backward_grads(); }
	GPU->dot(next->bias_activations, next->error,b_grad_next);
}


void Layer::weight_update()
{
	if(target){ return; }

	next->weight_update();

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



Layer *Layer::get_root()
{
	Layer *root = this;
	while(root->next){ root = root->next; }
	return root;
}


Layer::~Layer()
{
	cout << "destruct" << endl;
}


