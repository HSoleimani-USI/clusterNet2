#include <Timer.cuh>
#include <assert.h>
#include <stdio.h>
#include <iostream>

using std::endl;
using std::cout;




void Timer::tick()
{
	tick("default");
}
void Timer::tick(std::string name)
{
	if (m_dictTickTock.count(name) > 0)
	{
		if (m_dictTickTockCumulative.count(name) > 0)
		{
			m_dictTickTockCumulative[name] += tock(m_dictTickTock[name],
					0.0f);
			m_dictTickTock.erase(name);
		} else
		{
			m_dictTickTockCumulative[name] = tock(m_dictTickTock[name], 0.0f);
			m_dictTickTock.erase(name);
		}
	} else
	{
		m_dictTickTock[name] = create_tick();
	}
}


float Timer::tock(){ return tock("default"); }
float Timer::tock(std::string name)
{
	if (m_dictTickTockCumulative.count(name) > 0)
	{
		tock("<<<Cumulative>>>: " + name, m_dictTickTockCumulative[name]);
		float value = m_dictTickTockCumulative[name];
		m_dictTickTockCumulative.erase(name);
		return value;
	}
	else
	{
		if (m_dictTickTock.count(name) == 0)
			cout << "Error for name: " << name << endl;
		assert(("No tick event was registered for the name" + name, m_dictTickTock.count(name) > 0));
		float value = tock(m_dictTickTock[name], name);
		m_dictTickTock.erase(name);
		return value;
	}
}




cudaEvent_t* Timer::create_tick()
{
    cudaEvent_t* startstop;
    startstop = (cudaEvent_t*)malloc(2*sizeof(cudaEvent_t));
    cudaEventCreate(&startstop[0]);
    cudaEventCreate(&startstop[1]);
    cudaEventRecord(startstop[0], 0);

    return startstop;
}

float Timer::tock(cudaEvent_t* startstop){ return tock(startstop, "Time for the kernel(s): "); }
float Timer::tock(cudaEvent_t* startstop, std::string text)
{
	float time;
	cudaEventRecord(startstop[1], 0);
	cudaEventSynchronize(startstop[1]);
	cudaEventElapsedTime(&time, startstop[0], startstop[1]);
	printf((text + ": %f ms.\n").c_str(), time);
	return time;
}
float Timer::tock(std::string text, float tocks)
{
	printf((text + ": %f ms.\n").c_str(), tocks);
	return tocks;
}

float Timer::tock(cudaEvent_t* startstop, float tocks)
{
	float time;
	cudaEventRecord(startstop[1], 0);
	cudaEventSynchronize(startstop[1]);
	cudaEventElapsedTime(&time, startstop[0], startstop[1]);

	return time+tocks;
}
