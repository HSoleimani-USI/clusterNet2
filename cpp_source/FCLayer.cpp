#include <FCLayer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

FCLayer::FCLayer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu, Network *network){ _LayerType = "FC"; init(unitcount, start_batch_size,unit,gpu, network); }
FCLayer::FCLayer(int unitcount, Unittype_t unit){ _LayerType = "FC"; init(unitcount, 0,unit, NULL, NULL); }
FCLayer::FCLayer(int unitcount){_LayerType = "FC";  init(unitcount, 0,Rectified_Linear, NULL, NULL); }


FCLayer::FCLayer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu)
{ _LayerType = "FC"; init(unitcount, start_batch_size,unit,gpu, NULL); prev->link_with_next_Layer(this); }
FCLayer::FCLayer(int unitcount, Unittype_t unit, Layer *prev){ _LayerType = "FC"; init(unitcount, 0,unit, prev->GPU, NULL); prev->link_with_next_Layer(this); }
FCLayer::FCLayer(int unitcount, Layer *prev){_LayerType = "FC";  init(unitcount, 0,Rectified_Linear, NULL, NULL); prev->link_with_next_Layer(this); }



void FCLayer::link_with_next_Layer(Layer *next_FCLayer)
{


	next = next_FCLayer;
	next->prev = this;
	next->Layer_ID = Layer_ID + 1;
	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->GPU){next->GPU = GPU;}
	if(!next->_network){next->_network = _network;}


	w_next = GPU->normal(UNITCOUNT,next_FCLayer->UNITCOUNT,0.0f,0.01f);;
	w_rms_next = zeros<float>(UNITCOUNT,next_FCLayer->UNITCOUNT);
	w_grad_next = zeros<float>(UNITCOUNT,next_FCLayer->UNITCOUNT);

	b_next = zeros<float>(1,next_FCLayer->UNITCOUNT);
	b_grad_next = zeros<float>(1,next_FCLayer->UNITCOUNT);
	b_rms_next = zeros<float>(1,next_FCLayer->UNITCOUNT);
}




void FCLayer::unit_activation()
{

	switch(UNIT_TYPE)
	{
		case Logistic:
			elementWiseUnary<klogistic>(activation, activation, 0.0f);
			break;
		case Rectified_Linear:
			elementWiseUnary<krectified>(activation, activation, 0.0f);
			break;
		case Softmax:
			softmax(activation,activation);
			break;
		case Linear:
			elementWiseUnary<kcopy>(activation, activation,0.0f);
			break;
		case Input:
			break;
	}

}

void FCLayer::activation_gradient()
{

	switch(UNIT_TYPE)
	{
		case Logistic:
			elementWiseUnary<klogistic>(activation, activation, 0.0f);
			break;
		case Rectified_Linear:
			elementWiseUnary<krectified_grad>(activation, activation, 0.0f);
			break;
		case Softmax:
			break;
		default:
			throw "Unknown unit";
			break;
	}

}

void FCLayer::forward()
{
	//cout << "pre init" << endl;
	//cout << BATCH_SIZE << " " << UNITCOUNT << endl;
	if(UNIT_TYPE != Input && (!activation))
	{
		bias_activations = ones<float>(1, BATCH_SIZE);
		activation = zeros<float>(BATCH_SIZE, UNITCOUNT);
		error = zeros<float>(BATCH_SIZE, UNITCOUNT);
	}

	if(!prev){ next->forward(); return; }

	//cout << "post init" << endl;

	//cout << "dropout" << endl;
	//cout << "pre dot" << endl;
	//cout << prev->w_next<< endl;
	//cout << activation<< endl;
	GPU->dot(prev->activation,prev->w_next,activation);

	//cout << "add vec" << endl;
	vectorWise<kvadd>(activation, prev->b_next, activation, 0.0f);
	//cout << "activate" << endl;
    unit_activation();

    //cout << "drop" << endl;
    //cout << "post drop" << endl;
    if(next){ next->forward(); }
}


void FCLayer::backward_errors()
{
    //cout << "errors" << endl;
	if(!target){ next->backward_errors(); }
	if(target)
	{
	    //cout << "target" << endl;
		if(activation->cols != target->cols && !target_matrix){ target_matrix = zeros<float>(BATCH_SIZE,activation->cols); }
		if(activation->cols != target->cols)
		{
			vectorWise<ktmatrix>(target,target, target_matrix,0.0f);
			elementWise<ksub>(activation,target_matrix,error,0.0f); return;
		}
		else{ elementWise<ksub>(activation,target,error,0.0f);  return;}


		elementWiseUnary<ksmul>(activation,activation,1.0f/error->rows); return;

	}


	if(UNIT_TYPE == Input){  return; }

	activation_gradient();
	//cout << "pre dotT" << endl;
	GPU->dotT(next->error, w_next,error);
	//cout << "pre elementwise" << endl;
	elementWise<kmul>(error, activation, error,0.0f);

}

void FCLayer::backward_grads()
{
	//cout << "grads" << endl;
	//cout << activation << " "<< next->error << " "<< w_grad_next << " "  << next->bias_activations << endl;
	GPU->Tdot(activation, next->error, w_grad_next);
	if(!next->target){ next->backward_grads(); }
	GPU->dot(next->bias_activations, next->error,b_grad_next);

}





