#ifndef SWAPTIONS_KNOBS_PRECISION_H
#define SWAPTIONS_KNOBS_PRECISION_H

#define MAX_TRIALS 102400

namespace Knobs 
{
    unsigned int GetNumTrials(unsigned int precision);
}

#endif //SWAPTIONS_KNOBS_PRECISION_H
