#include <Layer.h>
#include <ActivationFunction.h>
#include <Transformer.h>
#include <Configurator.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;


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

	_conf = new Configurator();
	_func = new ActivationFunction(unitType);

	if( unitType == Softmax)
		_transformer = NULL;
	else
		_transformer = new Transformer(DropoutTransform, this);


}


Matrix<float> *Layer::get_forward_activation()
{
	if(_transformer)
	{
		if(_transformer->_ttype == DropoutTransform)
		{
			return _transformer->output;
		}
	}

	return activation;
}

