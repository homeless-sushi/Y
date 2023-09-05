#ifndef NBODY_NBODY_NBODY_H
#define NBODY_NBODY_NBODY_H

#include "Nbody/Body.h"

#include <vector>

#define NBODY_SOFTENING 1e-9f

namespace Nbody 
{
    class Nbody 
    {
        public:
            virtual void run() = 0;
            float getSimulatedTime() { return simulatedTime; } 
            virtual std::vector<Body> getResult() = 0;
            
            virtual ~Nbody() = default;

        protected:

            Nbody(float simulationTime, float timeStep) :
                simulationTime{simulationTime},
                timeStep{timeStep}
            {};

            float simulationTime = 0;
            float simulatedTime = 0;
            float timeStep;
    };
}

#endif //NBODY_NBODY_NBODY_H