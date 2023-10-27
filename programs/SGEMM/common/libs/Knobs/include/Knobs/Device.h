#ifndef SGEMM_KNOBS_DEVICE
#define SGEMM_KNOBS_DEVICE

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
            enum TILE_SIZE
            {
                TILE_8 = 8,
                TILE_16 = 16,
            };
            TILE_SIZE tileSize;

            GpuKnobs(TILE_SIZE tileSize);
    };

    unsigned int GetCpuTileSizeFromExponent(unsigned int exp);
    GpuKnobs::TILE_SIZE GetGpuTileSizeFromExponent(unsigned int exp);
}

#endif //SGEMM_KNOBS_DEVICE

