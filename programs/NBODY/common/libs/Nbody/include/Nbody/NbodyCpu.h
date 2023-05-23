#ifndef NBODY_NBODY_NBODY_CPU_H
#define NBODY_NBODY_NBODY_CPU_H

#include "Nbody/Body.h"
#include "Nbody/Nbody.h"

#include <vector>

namespace NbodyCpu
{
    class NbodyCpu : public Nbody::Nbody
    {
        public:
            void run() override;
            std::vector<::Nbody::Body> getResult() override;
            
            NbodyCpu(
                const std::vector<::Nbody::Body>& bodies, 
                float timeStep,
                unsigned int nThreads
            );
        private:
            float dt;
            std::vector<::Nbody::Body> bodies;
            unsigned int nThreads;
    };
}

#endif //NBODY_NBODY_NBODY_CPU_H
