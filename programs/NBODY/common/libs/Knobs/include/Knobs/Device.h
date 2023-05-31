#ifndef NBODY_KNOBS_DEVICE_H
#define NBODY_KNOBS_DEVICE_H

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
            unsigned int N_THREADS;

            CpuKnobs(unsigned int N_THREADS);
    };

    class GpuKnobs : public DeviceKnobs
    {   
        public:
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

            GpuKnobs(BLOCK_SIZE BLOCK_SIZE);
    };
}

#endif //NBODY_KNOBS_DEVICE_H

