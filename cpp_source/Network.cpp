#include <Network.h>
#include <Layer.h>
#include <BatchAllocator.h>
#include <ErrorHandler.h>
#include <Optimizer.h>
#include <Configurator.h>
#include <Transformer.h>


Network::Network(ClusterNet *gpu)
{
	_isTrainTime = true;
	_gpu = gpu;
	_errorhandler = new ErrorHandler();
	_conf = new Configurator();
}

void Network::add(Layer *layer)
{
	Layer *prev;
	if(_layers.size() > 0)
	{
		prev = _layers.back();
		layer->prev = prev;
		prev->next = layer;
	}

	_layers.push_back(layer);
	layer->GPU = _gpu;
	layer->_network = this;
	if(layer->_transformer){ layer->_transformer->_gpu = _gpu; }
	if(layer->_transformer){ layer->_transformer->_net = this; }

}

void Network::init_weights(WeightInitType_t wtype)
{
	Layer *prev = NULL;
	for(int i = 0; i < _layers.size(); i++)
	{
		if(!prev){ prev = _layers[i]; continue; }

		if(wtype == Gaussian)
		{
			prev->w_next = _gpu->normal(prev->UNITCOUNT,_layers[i]->UNITCOUNT,0.0f,0.0001f);
		}
		else if(wtype == UniformSqrt)
		{
			prev->w_next = _gpu->rand(prev->UNITCOUNT,_layers[i]->UNITCOUNT);
			float range = 8.0f*sqrtf(6.0f/((float)prev->UNITCOUNT + _layers[i]->UNITCOUNT));
			elementWise<ksmul>(prev->w_next,prev->w_next,range);
			elementWise<kssub>(prev->w_next,prev->w_next,range/2.0f);
		}

		prev->w_rms_next = zeros<float>(prev->UNITCOUNT,_layers[i]->UNITCOUNT);
		prev->w_grad_next = zeros<float>(prev->UNITCOUNT,_layers[i]->UNITCOUNT);

		prev->b_next = zeros<float>(1,_layers[i]->UNITCOUNT);
		prev->b_grad_next = zeros<float>(1,_layers[i]->UNITCOUNT);
		prev->b_rms_next = zeros<float>(1,_layers[i]->UNITCOUNT);

		prev = _layers[i];
	}

}

void Network::init_activations(int batchsize)
{
	if(_layers.front()->_transformer){_layers.front()->_transformer->output = zeros<float>(batchsize, _layers.front()->UNITCOUNT);}

	for(int i = 1; i < _layers.size(); i++)
	{
		if(_layers[i]->bias_activations != NULL && _layers[i]->bias_activations->rows == batchsize){ return; }

		if(_layers[i]->bias_activations != NULL)
		{
			_layers[i]->bias_activations->free_matrix();
			_layers[i]->activation->free_matrix();
			_layers[i]->activation_grad->free_matrix();
			_layers[i]->error->free_matrix();
			if(i == _layers.size()-1){ _layers[i]->target_matrix->free_matrix(); }
			if(_layers[i]->_transformer){ _layers[i]->_transformer->output->free_matrix(); }
		}

		_layers[i]->bias_activations = ones<float>(1, batchsize);
		_layers[i]->activation = zeros<float>(batchsize, _layers[i]->UNITCOUNT);
		_layers[i]->activation_grad = zeros<float>(batchsize, _layers[i]->UNITCOUNT);
		_layers[i]->error = zeros<float>(batchsize, _layers[i]->UNITCOUNT);
		if(i == _layers.size()-1){ _layers[i]->target_matrix = zeros<float>(batchsize, _layers[i]->UNITCOUNT);}
		if(_layers[i]->_transformer){ _layers[i]->_transformer->output = zeros<float>(batchsize, _layers[i]->UNITCOUNT); }
	}

}

void Network::copy_global_params_to_layers()
{
	for(int i = 0; i < _layers.size(); i++)
	{
		_layers[i]->_conf->LEARNING_RATE = _conf->LEARNING_RATE;
		_layers[i]->_conf->RMSPROP_MOMENTUM = _conf->RMSPROP_MOMENTUM;
		_layers[i]->_conf->DROPOUT = _conf->DROPOUT;
	}
}

void Network::fit_partial(BatchAllocator *b, int batches)
{
	//cout << "init" << endl;
	init_activations(b->BATCH_SIZE);
	_isTrainTime = true;

	for(int i = 0; i < batches; i++)
	{
		//cout << "replace" << endl;
		b->replace_current_with_next_batch();
		//cout << "allocate" << endl;
		b->allocate_next_batch_async();
		//cout << "front" << endl;
		//cout << _layers.size() << endl;
		//cout << _layers.front() << endl;

		_layers.front()->activation = b->get_current_batchX();
		//cout << "back" << endl;
		_layers.back()->target = b->get_current_batchY();

		//cout << "forward" << endl;
		_layers.front()->forward();
		//cout << "add error" << endl;
		_errorhandler->add_error(_layers.back()->activation, _layers.back()->target);
		//cout << "errors" << endl;
		_layers.front()->backward_errors();
		//cout << "grads" << endl;
		//_layers.front()->backward_grads();

		//cout << "weight update" << endl;
		for(int j = 0; j < _layers.size()-1; j++)
		{
			_opt->weight_update(_layers[j]->w_rms_next, _layers[j]->w_next,_layers[j]->w_grad_next,_conf->RMSPROP_MOMENTUM,_conf->LEARNING_RATE);
			_opt->weight_update(_layers[j]->b_rms_next, _layers[j]->b_next,_layers[j]->b_grad_next,_conf->RMSPROP_MOMENTUM,_conf->LEARNING_RATE/100.0f);
		}
	}
	_errorhandler->print_error("Running train error: ");
}

void Network::fit(BatchAllocator *b, int epochs)
{
	for(int epoch = 0; epoch < epochs; epoch++)
	{
		cout << "EPOCH: " << epoch + 1 << endl;

		fit_partial(b, b->BATCHES);
		b->replace_current_with_next_batch();
		b->allocate_next_batch_async();
	}
}

void Network::train(BatchAllocator *train, BatchAllocator *CV, int epochs)
{
	for(int epoch = 0; epoch < epochs; epoch++)
	{
		cout << "EPOCH: " << epoch + 1 << endl;

		fit_partial(train, train->BATCHES);
		train->replace_current_with_next_batch();
		train->allocate_next_batch_async();

		get_errors(train, "Train error: ");
		get_errors(CV, "CV error: ");
	}
}

Matrix<float> *Network::predict(Matrix<float> *X)
{
	_isTrainTime = false;
	init_activations(X->rows);

	_layers.front()->activation = X;

	_layers.front()->forward();

	return _layers.back()->activation;
}

void Network::get_errors(BatchAllocator *b, std::string message)
{
	init_activations(b->BATCH_SIZE);
	_isTrainTime = false;
	for(int i = 0; i < b->BATCH_SIZE; i++)
	{
		b->replace_current_with_next_batch();
		b->allocate_next_batch_async();
		_layers.front()->activation = b->get_current_batchX();
		_layers.back()->target = b->get_current_batchY();

		_layers.front()->forward();

		_errorhandler->add_error(_layers.back()->activation, _layers.back()->target);
	}
	b->replace_current_with_next_batch();
	b->allocate_next_batch_async();
	_errorhandler->print_error(message);
}
