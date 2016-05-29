#include <Network.h>
#include <Layer.h>
#include <BatchAllocator.h>
#include <ErrorHandler.h>
#include <Optimizer.h>
#include <Configurator.h>
#include <Transformer.h>
#include <ActivationFunction.h>
#include <mph.h>


Network::Network(ClusterNet *gpu)
{
	_isTrainTime = true;
	_gpu = gpu;
	_errorhandler = new ErrorHandler(gpu);
	_conf = new Configurator();
	_opt = new Optimizer(gpu, RMSProp);
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
	layer->init_transformers(_gpu, this);
	layer->_func->GPU = _gpu;

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
			if(Is_initialized)
			{
				prev->_ga->init_MPI();
				prev->_ga->init_Matrix(prev->w_next);
			}
		}
		else if(wtype == UniformSqrt)
		{
			prev->w_next = _gpu->get_uniformsqrt_weight(prev->UNITCOUNT,_layers[i]->UNITCOUNT);
			if(Is_initialized)
			{
				prev->_ga->init_MPI();
				prev->_ga->init_Matrix(prev->w_next);
			}
		}

		prev->w_rms_next = _gpu->OPS->zeros(prev->UNITCOUNT,_layers[i]->UNITCOUNT);
		prev->w_grad_next = _gpu->OPS->zeros(prev->UNITCOUNT,_layers[i]->UNITCOUNT);

		prev->b_next = _gpu->OPS->zeros(1,_layers[i]->UNITCOUNT);
		prev->b_grad_next = _gpu->OPS->zeros(1,_layers[i]->UNITCOUNT);
		prev->b_rms_next = _gpu->OPS->zeros(1,_layers[i]->UNITCOUNT);


		prev = _layers[i];
	}

}

void Network::init_activations(int batchsize)
{
	_layers.front()->init_transformer_activations(batchsize);

	for(int i = 1; i < _layers.size(); i++)
	{
		_layers[i]->init_transformer_activations(batchsize);
		
		if(_layers[i]->activation != NULL && _layers[i]->activation->rows == batchsize){ return; }
		if(_layers[i]->activation != NULL)
		{
			_gpu->OPS->free_matrix(_layers[i]->activation);
			_gpu->OPS->free_matrix(_layers[i]->activation_grad);
			_gpu->OPS->free_matrix(_layers[i]->error);
			if(i == _layers.size()-1){ _gpu->OPS->free_matrix(_layers[i]->target_matrix); }
		}

		_layers[i]->activation = _gpu->OPS->zeros(batchsize, _layers[i]->UNITCOUNT);
		_layers[i]->activation_grad = _gpu->OPS->zeros(batchsize, _layers[i]->UNITCOUNT);
		_layers[i]->error = _gpu->OPS->zeros(batchsize, _layers[i]->UNITCOUNT);
		if(i == _layers.size()-1){ _layers[i]->target_matrix = _gpu->OPS->zeros(batchsize, _layers[i]->UNITCOUNT);}
	}

}

void Network::copy_global_params_to_layers()
{
	for(int i = 0; i < _layers.size(); i++)
	{
		_layers[i]->_conf->LEARNING_RATE = _conf->LEARNING_RATE;
		_layers[i]->_conf->RMSPROP_MOMENTUM = _conf->RMSPROP_MOMENTUM;
		_layers[i]->_conf->DROPOUT = _conf->DROPOUT;
		_layers[i]->_conf->LEARNING_RATE_DECAY = _conf->LEARNING_RATE_DECAY;
	}
}

void Network::fit_partial(BatchAllocator *b, int batches)
{
	
	init_activations(b->BATCH_SIZE);
	_isTrainTime = true;

	for(int i = 0; i < batches; i++)
	{
		cout << "current batchno: " << i << endl;
		//cout << "pre alloc" << endl;	
		b->replace_current_with_next_batch();
		b->allocate_next_batch_async();

		//cout << "sum batch: " << _gpu->OPS->sum(b->get_current_batchX()) << endl;
		//cout << _gpu->OPS->sum(b->get_current_batchX()) << endl;
		cout << "w sum: " << _gpu->OPS->sum(_layers.front()->w_next) << endl;

		_layers.front()->activation = b->get_current_batchX();
		_layers.back()->target = b->get_current_batchY();
		//cout << "pre forward" << endl;	
		_layers.front()->forward();
		_errorhandler->add_error(_layers.back()->activation, _layers.back()->target);
		//cout << "pre backward" << endl;	
		_layers.front()->backward_errors();

		//cout << "pre update" << endl;	
		for(int j = 0; j < _layers.size()-1; j++)
		{
			_layers[j]->_ga->send_MPI();
			_layers[j]->_ga->recv_MPI();
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
		//TODO: Why was this here?
		//train->replace_current_with_next_batch();
		//train->allocate_next_batch_async();

		get_errors(train, "Train error: ");
		get_errors(CV, "CV error: ");

		_conf->LEARNING_RATE *= _conf->LEARNING_RATE_DECAY;
		for(int i = 0; i < _layers.size(); i++)
			_layers[i]->_conf->LEARNING_RATE *= _layers[i]->_conf->LEARNING_RATE_DECAY;

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
