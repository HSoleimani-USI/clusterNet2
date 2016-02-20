/*
 * LookupLayer.h
 *
 *  Created on: Feb 19, 2016
 *      Author: tim
 */

#include <Layer.h>
#include <map>
#include <string>


#ifndef LOOKUPLAYER_H_
#define LOOKUPLAYER_H_


class LookupLayer : public Layer
{
public:
	LookupLayer(){};
	LookupLayer(int unitcount, std::map<std::string, int> vocab2idx);
	LookupLayer(int unitcount, std::map<std::string,int> vocab2idx, Matrix<float> *embeddings);
	~LookupLayer(){};



	void init_embeddings();
	void init_embeddings(Matrix<float> *embeddings);


	void forward();
	void backward_errors();
	void backward_grads();
	void update_embeddings();
private:
	std::map<std::string, int> _vocab2idx;
	Matrix<float> *_embeddings;
	Matrix<float> *_rms_embedding;
};

#endif /* LOOKUPLAYER_H_ */
