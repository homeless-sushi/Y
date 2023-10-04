#ifndef HISTO_HISTO_HISTO_CUDA_H
#define HISTO_HISTO_HISTO_CUDA_H

#include <Histo/Histo.h>

#include <vector>

#include <cuda_runtime_api.h>

namespace Histo 
{
    class HistoCuda : public ::Histo::Histo
    {
        public:
            HistoCuda(
                std::vector<unsigned short> rgb,
                unsigned blockSize,
                unsigned nBlocks
            );
            ~HistoCuda();
            void run();

            std::vector<unsigned> getResult();
            
            float getDataUploadTime() { return dataUploadTime; }
            float getKernelTime() { return kernelTime; }
            float getDataDownloadTime() { return dataDownloadTime; }

        private:
            unsigned short* rgb_device;
            unsigned* histo_partial_device;
            unsigned* histo_final_device;

            float dataUploadTime = 0;
            float kernelTime = 0;
            float dataDownloadTime = 0;

            unsigned blockSize;
            unsigned nBlocks;
    };
}

#endif //HISTO_HISTO_HISTO_CUDA_H
