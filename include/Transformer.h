/*
 * Transformer.h
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <ClusterNet.h>

class Layer;
class Network;

class Transformer
{
public:
	Transformer(TransformerType_t ttype, Layer *layer);
	~Transformer(){};

	Matrix<float> *output;

	Matrix<float> *transform(Matrix<float> *input);

	TransformerType_t _ttype;
	ClusterNet *_gpu;
	Layer* _layer;
	Network *_net;
};

#endif /* TRANSFORMER_H_ */
