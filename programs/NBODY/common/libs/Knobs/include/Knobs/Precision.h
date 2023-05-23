#ifndef NBODY_KNOBS_PRECISION_H
#define NBODY_KNOBS_PRECISION_H

namespace Knobs 
{
    float GetApproximateTimeStep(
        float simTime,
        float timeStep,
        unsigned int precision
    );

    float GetApproximateSimTime(
        float simTime,
        float timeStep,
        unsigned int precision
    );
}

#endif //NBODY_KNOBS_PRECISION_H
