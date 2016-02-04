#include <Layer.h>

#ifndef FCLayer_H
#define FCLayer_H

class FCLayer : public Layer
{
public:
	FCLayer(){};
	~FCLayer(){};

	FCLayer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu);
	FCLayer(int unitcount, Unittype_t unit);
	FCLayer(int unitcount);

	FCLayer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu);
	FCLayer(int unitcount, Unittype_t unit, Layer *prev);
	FCLayer(int unitcount, Layer *prev);

	void forward();
	void forward(bool useDropout);
	void backward_errors();
	void backward_grads();

	void link_with_next_Layer(Layer *next_Layer);


protected:
	void unit_activation();
	void unit_activation(bool useDropout);
	void activation_gradient();
	void apply_dropout();


};

#endif
