#include <Knobs/Device.h>

#include <cmath>

namespace Knobs 
{
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
