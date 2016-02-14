/*
 * Configurator.cpp
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#include "Configurator.h"

Configurator::Configurator()
{
	// TODO Auto-generated constructor stub
	LEARNING_RATE = 0.001f;
	RMSPROP_MOMENTUM = 0.9f;
	DROPOUT = 0.5f;

	LEARNING_RATE_DECAY = 0.995f;
}

Configurator::~Configurator() {
	// TODO Auto-generated destructor stub
}

