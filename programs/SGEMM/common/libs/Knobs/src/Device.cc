#include <Knobs/Device.h>

#include <string>
#include <unordered_map>

namespace Knobs 
{
    std::string DeviceToString(DEVICE device)
    {
        std::unordered_map<DEVICE, std::string> map;
        map[DEVICE::CPU] = std::string("CPU");
        map[DEVICE::GPU] = std::string("GPU");
        return map[device]; 
    }

    DeviceKnobs::DeviceKnobs(
        DEVICE device
    ) :
        device{device}
    {};

    CpuKnobs::CpuKnobs(
        unsigned int nThreads,
        unsigned int tileSize
    ) : 
        nThreads{nThreads},
        tileSize{tileSize},
        DeviceKnobs(DEVICE::CPU) 
    {};


    GpuKnobs::GpuKnobs(
        GpuKnobs::TILE_SIZE tileSize
    ) : 
        tileSize{tileSize},
        DeviceKnobs(DEVICE::GPU) 
    {};

    unsigned int GetCpuTileSizeFromExponent(unsigned int exp)
    {
        return 8 << exp;
    };
    GpuKnobs::TILE_SIZE GetGpuTileSizeFromExponent(unsigned int exp)
    {
        return static_cast<Knobs::GpuKnobs::TILE_SIZE>(
            static_cast<unsigned int>(Knobs::GpuKnobs::TILE_8) << exp
        );
    };
}
