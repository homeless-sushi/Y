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
            virtual std::vector<Body> getResult() = 0;
            
            virtual ~Nbody() = default;
    };
}

#endif //NBODY_NBODY_NBODY_H