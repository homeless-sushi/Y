#ifndef NBODY_NBODY_NBODY_CUDA_H
#define NBODY_NBODY_NBODY_CUDA_H

#include "Nbody/Body.h"
#include "Nbody/Nbody.h"

#include <utility> 
#include <vector>

namespace NbodyCuda 
{
    class BodySoa 
    {
        public: 
            float* x;
            float* y;
            float* z;
            float* vx;
            float* vy;
            float* vz;

            unsigned long n;

            BodySoa();
            BodySoa(unsigned long n);
            BodySoa(const std::vector<::Nbody::Body>& bodies);

            BodySoa(const BodySoa& owner);
            BodySoa& operator=(const BodySoa& other) = delete;

            BodySoa(BodySoa&& other) = delete;
            BodySoa& operator=(BodySoa&& other) = delete;

            ~BodySoa();

            void swap(BodySoa& other) {
                std::swap(x, other.x);
                std::swap(y, other.y);
                std::swap(z, other.z);
                std::swap(vx, other.vx);
                std::swap(vy, other.vy);
                std::swap(vz, other.vz);
                std::swap(n, other.n);
                std::swap(owner, other.owner);
            }

            std::vector<::Nbody::Body> getBodiesVector();

        private:
            bool owner;
    };

    class NbodyCuda : public Nbody::Nbody
    {
        public:
            void run() override;
            std::vector<::Nbody::Body> getResult() override;
            
            NbodyCuda(
                const std::vector<::Nbody::Body>& bodies,
                float timeStep,
                unsigned int blockSize
            );
            ~NbodyCuda() override;
        private:
            unsigned long n;
            float dt;
            BodySoa in;
            BodySoa out;

            unsigned int blockSize;
    };
}

#endif //NBODY_NBODY_NBODY_CUDA_H