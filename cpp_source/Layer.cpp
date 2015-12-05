#include <Layer.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

template <typename T> Layer<T>::Layer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<T> *gpu){ init(unitcount, start_batch_size,unit,gpu); }
template <typename T> Layer<T>::Layer(int unitcount, Unittype_t unit){ init(unitcount, 0,unit, NULL); }
template <typename T> Layer<T>::Layer(int unitcount){ init(unitcount, 0,Rectified_Linear, NULL); }

template <typename T> Layer<T>::Layer(int unitcount, int start_batch_size, Unittype_t unit, Layer<T> *prev, ClusterNet2<T> *gpu)
{ init(unitcount, start_batch_size,unit,gpu); prev->link_with_next_layer(this); }
template <typename T> Layer<T>::Layer(int unitcount, Unittype_t unit, Layer<T> *prev){ init(unitcount, 0,unit, prev->GPU); prev->link_with_next_layer(this); }
template <typename T> Layer<T>::Layer(int unitcount, Layer<T> *prev){ init(unitcount, 0,Rectified_Linear, NULL); prev->link_with_next_layer(this); }

template <typename T> void Layer<T>::init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<T> *gpu)
{

	next = NULL;
	prev = NULL;
	w_next = NULL;
	b_next = NULL;
	b_next_sync = NULL;
	w_rms_next = NULL;
	b_rms_next = NULL;
	b_grad_next = NULL;

	w_next_sync_send = NULL;
	b_next_sync_send = NULL;
	w_next_sync_recv = NULL;
	b_next_sync_recv = NULL;

	target = NULL;
	target_matrix = NULL;
	error = NULL;

	LEARNING_RATE = 0.003f;
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
		out = zeros<T>(BATCH_SIZE, UNITCOUNT);
		bias_activations = ones<T>(1, BATCH_SIZE);
		activation = zeros<T>(BATCH_SIZE, UNITCOUNT);
	}
	else
	{
		out = NULL;
		bias_activations = NULL;
		activation = NULL;
	}

}

template <typename T> void Layer<T>::link_with_next_layer(Layer<T> *next_layer)
{

	next = next_layer;
	next->LAYER_ID = LAYER_ID + 1;
	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->GPU){next->GPU = GPU;}

	Matrix<T> *w = GPU->uniformSqrtWeight(UNITCOUNT,next_layer->UNITCOUNT);
	//Matrix<T> *w = GPU->randn(UNITCOUNT,next_layer->UNITCOUNT,0,0.001);
	w_next = w;
	w_rms_next = zeros<T>(UNITCOUNT,next_layer->UNITCOUNT);
	for(int i = 0; i < GPU->MPI_SIZE; i++) vec_w_grad_next.push_back(zeros<T>(UNITCOUNT,next_layer->UNITCOUNT));

	Matrix<T> *b = zeros<T>(1,next_layer->UNITCOUNT);
	b_next = b;
	b_grad_next = zeros<T>(1,next_layer->UNITCOUNT);
	b_rms_next = zeros<T>(1,next_layer->UNITCOUNT);

	next->out = zeros<T>(BATCH_SIZE, next->UNITCOUNT);
	next->activation = zeros<T>(BATCH_SIZE, next->UNITCOUNT);
	next->error = zeros<T>(BATCH_SIZE, next->UNITCOUNT);
	next->bias_activations = ones<T>(1, BATCH_SIZE);
	next->prev = this;
}


template <typename T> void Layer<T>::unit_activation(){ unit_activation(true); }
template <typename T> void Layer<T>::unit_activation(bool useDropout)
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
			//elementWiseUnary<(out, activation);
			break;
		case Input:
			break;
	}


	if(UNIT_TYPE != Softmax)
	{
		if(!useDropout)
			elementWiseUnary<ksmul>(activation,out,(T)(1.0f-DROPOUT));
	}



}

template <typename T> void Layer<T>::apply_dropout()
{
	if(UNIT_TYPE != Softmax)
		{
			GPU->dropout(activation,out,DROPOUT);
		}
}

template <typename T> void Layer<T>::activation_gradient()
{

	switch(UNIT_TYPE)
	{
		case Logistic:
			elementWiseUnary<klogistic>(activation, out, (T)0.0f);
			break;
		case Rectified_Linear:
			elementWiseUnary<krectified_grad>(activation, out, (T)0.0f);
			break;
		case Softmax:
			break;
		default:
			throw "Unknown unit";
			break;
	}

}

template <typename T> void Layer<T>::handle_offsize()
{
	if(!prev)
	{
		if(!out){ out = empty<T>(activation->rows, activation->cols); }
		else if(out->rows != activation->rows)
		{
			cudaFree(out->data);
			free(out);
			out = empty<T>(activation->rows, activation->cols);
		}
	}
	else
	{
		if(prev->out->rows != out->rows && (!out_offsize || out_offsize->rows != prev->out->rows))
		{
			if(out_offsize)
			{
				cudaFree(out_offsize->data);
				cudaFree(activation_offsize->data);
				cudaFree(error_offsize->data);
				cudaFree(bias_activations_offsize->data);
				cudaFree(target_matrix_offsize->data);
			}

			out_offsize = empty<T>(prev->out->rows, UNITCOUNT);
			activation_offsize = empty<T>(prev->out->rows, UNITCOUNT);
			error_offsize = empty<T>(prev->out->rows, UNITCOUNT);
			bias_activations_offsize = empty<T>(1,prev->out->rows);
			target_matrix_offsize = zeros<T>(prev->out->rows, UNITCOUNT);
		}


		if(prev->out->rows != out->rows)
		{
			/*
			Matrix<T> *swap;
			swap = out; out = out_offsize; out_offsize = swap;
			swap = activation; activation = activation_offsize; activation_offsize = swap;
			swap = error; error = error_offsize; error_offsize = swap;
			swap = bias_activations; bias_activations = bias_activations_offsize; bias_activations_offsize = swap;
			swap = target_matrix; target_matrix = target_matrix_offsize; target_matrix_offsize = swap;
			*/
		}
	}

}

template <typename T> void Layer<T>::forward(){ forward(true); }
template <typename T> void Layer<T>::forward(bool useDropout)
{
	handle_offsize();
	if(!prev){  unit_activation(useDropout); if(useDropout){apply_dropout(); } next->forward(useDropout); return; }
	if(useDropout){  prev->weight_update(); }



	GPU->dot(prev->out,prev->w_next,out);


	vectorWise<kvadd>(out, prev->b, out, 0.0f);
    unit_activation(useDropout);

    if(useDropout){apply_dropout(); }
    if(next){ next->forward(useDropout); }
}


template <typename T> void Layer<T>::running_error()
{
	if(!target){ next->running_error(); return;}

	string text = "";

	Matrix<T> *result;
	Matrix<T> *eq = empty<T>(target->rows, target->cols);

	float sum_value = 0.0f;

	switch(COST)
	{
		case Misclassification:
			//result = argmax(out);
			elementWise<keq>(result,target,eq);
			//sum_value = sum(eq);
			//RUNNING_ERROR += (out->rows  - sum_value);
			//RUNNING_SAMPLE_SIZE += out->rows;
			break;
		default:
			throw "Unknown cost function!";
			break;
	}

	cudaFree(result->data);
	cudaFree(eq->data);
}



template <typename T> void Layer<T>::backward_errors()
{
	if(!target){ next->backward_errors(); }
	if(target)
	{
		if(out->cols != target->cols && !target_matrix){ target_matrix = zeros<T>(BATCH_SIZE,out->cols); }
		//if(out->cols != target->cols){ create_t_matrix(target,target_matrix); sub(out,target_matrix,error); return; }
		else{ sub(activation,target,error);  return;}
	}

	if(UNIT_TYPE == Input){ backward_grads(); return; }

	activation_gradient();

	GPU->dotT(next->error, w_next,error);
	mul(error, out, error);

}

template <typename T> void Layer<T>::backward_grads()
{
	GPU->Tdot(activation, next->error, vec_w_grad_next[GPU->MYRANK]);
	if(!next->target){ next->backward_grads(); }
	GPU->dot(next->bias_activations, next->error,b_grad_next);

}


template <typename T> void Layer<T>::weight_update()
{
	if(target){ return; }

	//next->weight_update();

	switch(UPDATE_TYPE)
	{
		case RMSProp:
			//scalarMul(vec_w_grad_next[GPU->MYRANK],1.25,vec_w_grad_next[GPU->MYRANK]);
			//mean = sum(vec_w_grad_next[GPU->MYRANK])/float(vec_w_grad_next[GPU->MYRANK]->size);
			//cout << mean << endl;
			RMSprop_with_weight_update(w_rms_next,vec_w_grad_next[GPU->MYRANK],w_next,w_next,RMSPROP_MOMENTUM,LEARNING_RATE,out->rows*GPU->MPI_SIZE,MOMENTUM);
			//RMSprop_with_weight_update(b_rms_next,b_grad_next,b_next,b_next,RMSPROP_MOMENTUM,LEARNING_RATE,out->rows*GPU->MPI_SIZE,MOMENTUM);
			RMSprop_with_weight_update(b_rms_next,b_grad_next,b_next,b_next,RMSPROP_MOMENTUM,LEARNING_RATE/100.0f,out->rows,MOMENTUM);
			//scalarMul(b_grad_next, LEARNING_RATE/float(out->rows*GPU->MPI_SIZE) ,b_grad_next);
			//sub(b_next,b_grad_next,b_next);

			break;
		case PlainSGD:
			scalarMul(vec_w_grad_next[GPU->MYRANK],LEARNING_RATE,vec_w_grad_next[GPU->MYRANK]);
			sub(w_next,vec_w_grad_next[GPU->MYRANK],w_next);
			break;
		default:
			throw "Unknown update type!";
			break;
	}

	//limit_magnitude();

	//cudaFree(noise->data);
	//free(noise);

}

template <typename T> void Layer<T>::print_error(string message)
{
	if(!target){ next->print_error(message); return;}



		if(message == "Train error: ")
		{
			/*
			print_percentile(prev->prev->vec_w_grad_next[GPU->MYRANK]);
			print_percentile(prev->out);
			print_percentile(prev->activation);

			cout << getNonZeroElements(prev->out) << endl;
			cout << getNonZeroElements(prev->prev->out) << endl;
			*/
		}


		cout << message << RUNNING_ERROR/RUNNING_SAMPLE_SIZE << endl;


	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
}

template <typename T> void Layer<T>::set_hidden_dropout(float dropout)
{
	if(!next){ return; }
	next->DROPOUT = dropout;
	next->set_hidden_dropout(dropout);
}

template <typename T> void Layer<T>::learning_rate_decay(float decay_rate)
{
	if(!next){ return; }
	next->LEARNING_RATE *= decay_rate;
	next->learning_rate_decay(decay_rate);
}

template <typename T> void Layer<T>::dropout_decay()
{
	if(!prev){ cout << "Decaying dropout!" << endl; }
	if(!next){ return;}

	cout << "Setting dropout from " << DROPOUT << " to " << DROPOUT/2.0f << endl;
	DROPOUT /= 2.0f;
	next->dropout_decay();
}


template <typename T> Layer<T> *Layer<T>::get_root()
{
	Layer<T> *root = this;
	while(root->next){ root = root->next; }
	return root;
}

template <typename T> Layer<T>::~Layer<T>()
{
	cout << "destruct" << endl;
}


