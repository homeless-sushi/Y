#include "Knobs/Precision.h"

namespace Knobs 
{
    unsigned int GetNumTrials(unsigned int precision)
    {
        return (MAX_TRIALS/100) * precision;
    }
}

