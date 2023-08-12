#ifndef SWAPTIONS_GPU_KERNEL_H
#define SWAPTIONS_GPU_KERNEL_H

#include "Swaptions/Swaptions.h"

namespace SwaptionsGpu
{
    void kernel(
        parm *swaptions,
        unsigned int nSwaptions,
        unsigned int gpuBlockSize,
        unsigned int tileSize,
        unsigned int nTrials,
        long seed
    );
}

#endif //SWAPTIONS_GPU_KERNEL_H
