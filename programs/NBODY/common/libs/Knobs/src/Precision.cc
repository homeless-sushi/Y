#include <Knobs/Precision.h>

#include <cmath>

namespace Knobs 
{
    float GetApproximateTimeStep(
        float simTime,
        float timeStep,
        unsigned int precision
    )
    {
        return timeStep*(100.f/precision);
    };

    float GetApproximateSimTime(
        float simTime,
        float timeStep,
        unsigned int precision
    )
    {
        unsigned int iterations = std::ceil(simTime/timeStep);
        float approximateIterations = iterations*(precision/100.f);
        return approximateIterations*timeStep;
    };
}
