#include <FCLayer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

FCLayer::FCLayer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu){ init(unitcount, start_batch_size,unit,gpu); }
FCLayer::FCLayer(int unitcount, Unittype_t unit){ init(unitcount, 0,unit, NULL); }
FCLayer::FCLayer(int unitcount){ init(unitcount, 0,Rectified_Linear, NULL); }


FCLayer::FCLayer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu)
{ init(unitcount, start_batch_size,unit,gpu); prev->link_with_next_Layer(this); }
FCLayer::FCLayer(int unitcount, Unittype_t unit, Layer *prev){ init(unitcount, 0,unit, prev->GPU); prev->link_with_next_Layer(this); }
FCLayer::FCLayer(int unitcount, Layer *prev){ init(unitcount, 0,Rectified_Linear, NULL); prev->link_with_next_Layer(this); }



void FCLayer::link_with_next_Layer(Layer *next_FCLayer)
{

	next = next_FCLayer;
	next->prev = this;
	next->Layer_ID = Layer_ID + 1;
	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->GPU){next->GPU = GPU;}


	w_next = GPU->normal(UNITCOUNT,next_FCLayer->UNITCOUNT,0.0f,0.01f);;
	w_rms_next = zeros<float>(UNITCOUNT,next_FCLayer->UNITCOUNT);
	w_grad_next = zeros<float>(UNITCOUNT,next_FCLayer->UNITCOUNT);

	b_next = zeros<float>(1,next_FCLayer->UNITCOUNT);
	b_grad_next = zeros<float>(1,next_FCLayer->UNITCOUNT);
	b_rms_next = zeros<float>(1,next_FCLayer->UNITCOUNT);
}




void FCLayer::unit_activation(){ unit_activation(true); }
void FCLayer::unit_activation(bool useDropout)
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

void FCLayer::apply_dropout()
{
	if(UNIT_TYPE != Softmax)
	{
		GPU->dropout(activation,out,DROPOUT);
	}
}

void FCLayer::activation_gradient()
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

void FCLayer::forward(){ forward(true); }
void FCLayer::forward(bool useDropout)
{
	cout << "pre init" << endl;
	cout << UNIT_TYPE << " " << out << " " << activation << endl;
	cout << BATCH_SIZE << " " << UNITCOUNT << endl;
	if(UNIT_TYPE != Input && (!out || !activation))
	{
		out = zeros<float>(BATCH_SIZE, UNITCOUNT);
		bias_activations = ones<float>(1, BATCH_SIZE);
		activation = zeros<float>(BATCH_SIZE, UNITCOUNT);
		error = zeros<float>(BATCH_SIZE, UNITCOUNT);
	}
	else if(UNIT_TYPE == Input && !out)
	{
		out = zeros<float>(BATCH_SIZE, UNITCOUNT);
	}

	cout << "post init" << endl;

	cout << "dropout" << endl;
	if(!prev){  unit_activation(useDropout); if(useDropout){apply_dropout(); } next->forward(useDropout); return; }
	cout << "pre dot" << endl;
	cout << prev->out << endl;
	cout << prev->w_next<< endl;
	cout << out<< endl;
	GPU->dot(prev->out,prev->w_next,out);

	cout << "add vec" << endl;
	vectorWise<kvadd>(out, prev->b_next, out, 0.0f);
	cout << "activate" << endl;
    unit_activation(useDropout);

    cout << "drop" << endl;
    if(useDropout){apply_dropout(); }
    cout << "post drop" << endl;
    if(next){ next->forward(useDropout); }
}


void FCLayer::backward_errors()
{
    cout << "errors" << endl;
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

void FCLayer::backward_grads()
{
	GPU->Tdot(activation, next->error, w_grad_next);
	if(!next->target){ next->backward_grads(); }
	GPU->dot(next->bias_activations, next->error,b_grad_next);
}





