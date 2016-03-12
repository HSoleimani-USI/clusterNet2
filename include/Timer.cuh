#ifndef Timer_H
#define Timer_H
#include <BasicOpsCUDA.cuh>
#include <string>
#include <map>


struct Timer
{
	void tick(std::string name);
	void tick();
	float tock(std::string name);
	float tock();
	std::map<std::string,cudaEvent_t*> m_dictTickTock;
	std::map<std::string,float> m_dictTickTockCumulative;

	cudaEvent_t* create_tick();
	float tock(cudaEvent_t* startstop);
	float tock(cudaEvent_t* startstop, std::string text);
	float tock(std::string text, float tocks);
	float tock(cudaEvent_t* startstop, float tocks);
};

#endif
