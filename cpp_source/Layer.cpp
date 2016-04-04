#include <Layer.h>
#include <ActivationFunction.h>
#include <Transformer.h>
#include <Configurator.h>


using std::cout;
using std::endl;
using std::string;
using std::vector;


void Layer::init(int unitcount, Unittype_t unitType, Layer_t layerType)
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
	activation = NULL;

	target = NULL;
	target_matrix = NULL;
	error = NULL;

	UNITCOUNT = unitcount;
	_LayerType = layerType;

	Layer_ID = 0;

	_conf = new Configurator();
	_func = new ActivationFunction(unitType, GPU);

	if( unitType != Softmax)
		_transformer.push_back(new Transformer(DropoutTransform, this));


}


Matrix<float> *Layer::get_forward_activation()
{
	if(!_transformer.empty())
	{
		if(_transformer.back()->_ttype == DropoutTransform)
		{
			return _transformer.back()->output;
		}
	}

	return activation;
}

void Layer::apply_transformations()
{
	if(!_transformer.empty())
	{
		Matrix<float> *input = activation;
		for(int i = 0; i < _transformer.size(); i++)
		{
			_transformer[i]->transform(input);
			input = _transformer[i]->output;
		}
	}
}


void Layer::init_transformers(ClusterNet *gpu, Network *net)
{
	if(!_transformer.empty())
	{
		for(int i = 0; i < _transformer.size(); i++)
		{
			_transformer[i]->_gpu = gpu;
			_transformer[i]->_net = net;
		}
	}
}

void Layer::init_transformer_activations(int batch_size)
{

	for(int i = 0; i < _transformer.size(); i++)
	{
		if(_transformer[i]->output != NULL && _transformer[i]->output->rows == batch_size){return;}
		if(_transformer[i]->output != NULL){ GPU->OPS->free_matrix( _transformer[i]->output); }

		Matrix<float> *prev = NULL;
		if(_transformer[i]->_ttype == DropoutTransform)
		{
			if(prev)
				_transformer[i]->output = GPU->OPS->empty(batch_size, prev->cols);
			else
				_transformer[i]->output = GPU->OPS->empty(batch_size, UNITCOUNT);

			prev = _transformer[i]->output;
		}

	}
}

