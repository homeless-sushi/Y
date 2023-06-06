#include <Knobs/Device.h>

#include <string>
#include <unordered_map>

#include <cmath>

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
        unsigned int N_THREADS
    ) : 
        N_THREADS{N_THREADS},
        DeviceKnobs(DEVICE::CPU) 
    {};


    GpuKnobs::GpuKnobs(
        GpuKnobs::BLOCK_SIZE blockSize
    ) : 
        blockSize{blockSize},
        DeviceKnobs(DEVICE::GPU) 
    {};
}
