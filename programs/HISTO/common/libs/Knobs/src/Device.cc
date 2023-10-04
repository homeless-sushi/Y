#include "Knobs/Device.h"

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
        unsigned int nThreads
    ) : 
        nThreads{nThreads},
        DeviceKnobs(DEVICE::CPU) 
    {};


    GpuKnobs::GpuKnobs(
        GpuKnobs::N_BLOCKS nBlocks,
        GpuKnobs::BLOCK_SIZE blockSize
    ) : 
        nBlocks{nBlocks},
        blockSize{blockSize},
        DeviceKnobs(DEVICE::GPU) 
    {};


    GpuKnobs::N_BLOCKS nBlocksfromExponent(unsigned int exp){
        return static_cast<GpuKnobs::N_BLOCKS>( 8 * (2 << exp));
    };
    GpuKnobs::BLOCK_SIZE blockSizefromExponent(unsigned int exp){
        return static_cast<GpuKnobs::BLOCK_SIZE>( 32 * (2 << exp));
    };
}
