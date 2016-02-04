#ifndef DropoutLayer_H
#define DropoutLayer_H

#include <ClusterNet.h>
#include <basicOps.cuh>
#include <Layer.h>
#include <Network.h>

class DropoutLayer : public Layer
{
public:
	DropoutLayer(){};
	~DropoutLayer(){};

	DropoutLayer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu);
	DropoutLayer(int unitcount, Unittype_t unit);
	DropoutLayer(int unitcount);

	DropoutLayer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu);
	DropoutLayer(int unitcount, Unittype_t unit, Layer *prev);
	DropoutLayer(int unitcount, Layer *prev);

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
