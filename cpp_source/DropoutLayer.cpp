#include <DropoutLayer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

DropoutLayer::DropoutLayer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu){ _LayerType = "Dropout"; init(unitcount, start_batch_size,unit,gpu, NULL); }
DropoutLayer::DropoutLayer(int unitcount, Unittype_t unit){ _LayerType = "Dropout"; init(unitcount, 0,unit, NULL, NULL); }
DropoutLayer::DropoutLayer(int unitcount){ _LayerType = "Dropout"; init(unitcount, 0,Rectified_Linear, NULL, NULL); }


DropoutLayer::DropoutLayer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu)
{ _LayerType = "Dropout"; init(unitcount, start_batch_size,unit,gpu, NULL); prev->link_with_next_Layer(this); }
DropoutLayer::DropoutLayer(int unitcount, Unittype_t unit, Layer *prev){ _LayerType = "Dropout"; init(unitcount, 0,unit, prev->GPU, NULL); prev->link_with_next_Layer(this); }
DropoutLayer::DropoutLayer(int unitcount, Layer *prev){ _LayerType = "Dropout"; init(unitcount, 0,Rectified_Linear, NULL, NULL); prev->link_with_next_Layer(this); }



void DropoutLayer::link_with_next_Layer(Layer *next_FCLayer)
{

	next = next_FCLayer;
	next->prev = this;
	next->Layer_ID = Layer_ID + 1;

	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->GPU){next->GPU = GPU;}
	if(!next->_network){next->_network = _network;}

	w_next = prev->w_next;
	w_rms_next = prev->w_rms_next;
	w_grad_next = prev->w_grad_next;

	b_next = prev->b_next;
	b_grad_next = prev->b_grad_next;
	b_rms_next = prev->b_rms_next;


}

void DropoutLayer::unit_activation(){}
void DropoutLayer::activation_gradient(){}

void DropoutLayer::forward()
{



	if(UNIT_TYPE != Input && (!activation))
	{
		cout << "alloc" << endl;
		//bias_activations = ones<float>(1, BATCH_SIZE);
		activation = zeros<float>(BATCH_SIZE, prev->UNITCOUNT);
		//error = zeros<float>(BATCH_SIZE, UNITCOUNT);
	}


	//cout << BATCH_SIZE << " " << UNITCOUNT << endl;

	//cout << _network << endl;

	//cout << prev->activation << " " << activation << endl;
	//cout << next << endl;


	if(_network->_isTrainTime)
		GPU->dropout(prev->activation,activation,DROPOUT);
	else
		elementWiseUnary<ksmul>(prev->activation,activation,(1.0f-DROPOUT));



	//activation = prev->activation;
	bias_activations = prev->bias_activations;

    if(next){ next->forward(); }
}


void DropoutLayer::backward_errors()
{
	if(!target){ next->backward_errors(); }

	error = next->error;


}

void DropoutLayer::backward_grads()
{
	if(!next->target){ next->backward_grads(); }
	w_grad_next = next->w_grad_next;
}





