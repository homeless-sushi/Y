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
        unsigned int blockSize
    ) : 
        blockSize{blockSize},
        DeviceKnobs(DEVICE::GPU) 
    {};
}
