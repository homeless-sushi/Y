#include "Histo/Const.h"
#include "Histo/Histo.h"

#include <vector>

namespace Histo 
{
    Histo::Histo(
        std::vector<unsigned short> rgb
    ) : 
        rgb{rgb},
        histo(N_CHANNEL_VALUES*N_CHANNELS, 0)
    {};

}
