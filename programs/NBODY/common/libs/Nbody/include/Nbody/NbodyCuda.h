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

                std::swap(dataUploadTime, other.dataUploadTime);
                std::swap(dataDownloadTime, other.dataDownloadTime);
            }

            std::vector<::Nbody::Body> getBodiesVector();

            float getDataUploadTime() { return dataUploadTime; };
            float getDataDownloadTime() { return dataDownloadTime; };
            
        private:
            bool owner;

            float dataUploadTime;
            float dataDownloadTime;
    };

    class NbodyCuda : public Nbody::Nbody
    {
        public:
            void run() override;
            std::vector<::Nbody::Body> getResult() override;
            
            NbodyCuda(
                const std::vector<::Nbody::Body>& bodies,
                float simulationTime,
                float timeStep,
                unsigned int blockSize
            );
            ~NbodyCuda() override;

            float getDataUploadTime() { return dataUploadTime; }
            float getKernelTime() { return kernelTotalTime; }
            float getDataDownloadTime() { return dataDownloadTime; }

        private:
            unsigned long n;
            BodySoa in;
            BodySoa out;

            unsigned int blockSize;

            float dataUploadTime = 0;
            float kernelTotalTime = 0;
            float dataDownloadTime = 0;
    };
}

#endif //NBODY_NBODY_NBODY_CUDA_H