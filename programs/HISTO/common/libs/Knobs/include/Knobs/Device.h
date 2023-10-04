#ifndef HISTO_KNOBS_DEVICE_H
#define HISTO_KNOBS_DEVICE_H

#include <string>

namespace Knobs 
{
    enum DEVICE 
    {
        CPU,
        GPU
    };

    std::string DeviceToString(DEVICE device);

    struct DeviceKnobs
    {   
        public:
            const DEVICE device;
        
        protected:
            DeviceKnobs(DEVICE device);
    };

    class CpuKnobs : public DeviceKnobs
    {   
        public:
            unsigned int nThreads;

            CpuKnobs(unsigned int nThreads);
    };

    class GpuKnobs : public DeviceKnobs
    {   
        public:
            enum N_BLOCKS
            {
                BLOCKS_8 = 8,
                BLOCKS_16 = 16,
                BLOCKS_32 = 32,
                BLOCKS_64 = 64,
                BLOCKS_128 = 128,
                BLOCKS_256 = 256
            };
            N_BLOCKS nBlocks;

            enum BLOCK_SIZE
            {
                BLOCK_32 = 32,
                BLOCK_64 = 64,
                BLOCK_128 = 128,
                BLOCK_256 = 256,
                BLOCK_512 = 512,
                BLOCK_1024 = 1024
            };
            BLOCK_SIZE blockSize;

            GpuKnobs(N_BLOCKS nBlocks, BLOCK_SIZE blockSize);
    };

    GpuKnobs::N_BLOCKS nBlocksfromExponent(unsigned int exp);
    GpuKnobs::BLOCK_SIZE blockSizefromExponent(unsigned int exp);
}

#endif //HISTO_KNOBS_DEVICE_H
