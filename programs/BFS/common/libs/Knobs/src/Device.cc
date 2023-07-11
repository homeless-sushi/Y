#include "Knobs/Device.h"

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
        unsigned int N_THREADS
    ) : 
        N_THREADS{N_THREADS},
        DeviceKnobs(DEVICE::CPU) 
    {};

    GpuKnobs::GpuKnobs(
        GpuKnobs::BLOCK_SIZE BLOCK_SIZE,
        GpuKnobs::CHUNK_FACTOR CHUNK_FACTOR,
        GpuKnobs::MEMORY_TYPE EDGES_OFFSETS_MEM_TYPE,
        GpuKnobs::MEMORY_TYPE EDGES_MEM_TYPE
    ) : 
        blockSize{BLOCK_SIZE},
        chunkSize{CHUNK_FACTOR},
        edgesOffsets{EDGES_OFFSETS_MEM_TYPE},
        edges{EDGES_MEM_TYPE},
        DeviceKnobs(DEVICE::GPU) 
    {};
    
}
