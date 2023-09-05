#include "Nbody/Body.h"
#include "Nbody/Nbody.h"
#include "Nbody/NbodyCpu.h"

#include <vector>

namespace NbodyCpu
{
    void NbodyCpu::run()
    {
        for(simulatedTime = 0; simulatedTime < simulationTime; simulatedTime+=timeStep){

            #pragma omp parallel for num_threads(nThreads)
            for (auto current = bodies.begin(); current < bodies.end(); current++){

                float Fx = 0.0f; 
                float Fy = 0.0f; 
                float Fz = 0.0f;

                for (auto other = bodies.begin(); other < bodies.end(); other++) {
                    const float dx = other->pos.x - current->pos.x;
                    const float dy = other->pos.y - current->pos.y;
                    const float dz = other->pos.z - current->pos.z;

                    const float distSqr = dx*dx + dy*dy + dz*dz + NBODY_SOFTENING;
                    const float invDist = 1/sqrtf(distSqr);
                    const float invDist3 = invDist * invDist * invDist;

                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }

                current->vel.x += timeStep*Fx; 
                current->vel.y += timeStep*Fy; 
                current->vel.z += timeStep*Fz;
            }

            #pragma parallel omp for num_threads(nThreads)
            for (auto body = bodies.begin(); body < bodies.end(); body++){
                body->pos.x += body->vel.x*timeStep; 
                body->pos.y += body->vel.y*timeStep; 
                body->pos.z += body->vel.z*timeStep;
            }

        }
    }

    std::vector<::Nbody::Body> NbodyCpu::getResult() { return bodies; };

    NbodyCpu::NbodyCpu(
        const std::vector<::Nbody::Body>& bodies,
        float simulationTime,
        float timeStep,
        unsigned int nThreads
    ) :
        Nbody::Nbody(simulationTime, timeStep),
        bodies{bodies},
        nThreads{nThreads}
    {};
}
