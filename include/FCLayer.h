#include <Layer.h>
#include <ClusterNet.h>

#ifndef FCLayer_H
#define FCLayer_H

class FCLayer : public Layer
{
public:
	FCLayer(int unitcount, Unittype_t unitType);
	~FCLayer(){};

	void forward();
	void backward_errors();
	void backward_grads();



};

#endif
