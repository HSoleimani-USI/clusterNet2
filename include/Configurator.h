/*
 * Configurator.h
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#ifndef CONFIGURATOR_H_
#define CONFIGURATOR_H_

class Configurator
{
public:
	Configurator();
	virtual ~Configurator();

	float LEARNING_RATE;
	float RMSPROP_MOMENTUM;

	void set_hidden_dropout(float dropout);

	void dropout_decay();
	void learning_rate_decay(float decay_rate);
};

#endif /* CONFIGURATOR_H_ */
