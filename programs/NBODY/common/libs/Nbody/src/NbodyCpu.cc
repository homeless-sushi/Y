#include "Nbody/Body.h"
#include "Nbody/Nbody.h"
#include "Nbody/NbodyCpu.h"

#include <vector>

namespace NbodyCpu
{
    void NbodyCpu::run()
    {
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

            current->vel.x += dt*Fx; 
            current->vel.y += dt*Fy; 
            current->vel.z += dt*Fz;
        }

        #pragma parallel omp for num_threads(nThreads)
        for (auto body = bodies.begin(); body < bodies.end(); body++){
            body->pos.x += body->vel.x*dt; 
            body->pos.y += body->vel.y*dt; 
            body->pos.z += body->vel.z*dt;
        }

    }

    std::vector<::Nbody::Body> NbodyCpu::getResult() { return bodies; };

    NbodyCpu::NbodyCpu(
        const std::vector<::Nbody::Body>& bodies, 
        float timeStep,
        unsigned int nThreads
    ) :
        bodies{bodies},
        dt{timeStep},
        nThreads{nThreads}
    {};
}
