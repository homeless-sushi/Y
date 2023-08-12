#ifndef SWAPTIONS_KNOBS_DEVICE
#define SWAPTIONS_KNOBS_DEVICE

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
            unsigned int tileSize;

            CpuKnobs(unsigned int nThreads, unsigned int tileSize);
    };

    class GpuKnobs : public DeviceKnobs
    {   
        public:
            unsigned int blockSize;

            GpuKnobs(unsigned int blockSize);
    };
}

#endif //SGEMM_KNOBS_DEVICE

