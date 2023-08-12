#ifndef SWAPTIONS_CPU_KERNEL_H
#define SWAPTIONS_CPU_KERNEL_H

#include "Swaptions/Swaptions.h"

namespace SwaptionsCpu
{
    void kernel(
        parm *swaptions,
        unsigned int nSwaptions,
        unsigned int nThreads,
        unsigned int tileSize,
        unsigned int nTrials,
        long seed
    );
}

#endif //SWAPTIONS_CPU_KERNEL_H
