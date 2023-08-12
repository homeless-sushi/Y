#ifndef SWAPTIONS_SWAPTIONS_READWRITE_H
#define SWAPTIONS_SWAPTIONS_READWRITE_H

#include "Swaptions/Swaptions.h"

#include <string>

namespace Swaptions 
{
    void WriteSwaptionsPrices(
        const std::string fileURL,
        unsigned int nSimulations,
        unsigned int nSwaptions,
        parm* swaptions
    );
}

#endif //SWAPTIONS_SWAPTIONS_READWRITE_H
