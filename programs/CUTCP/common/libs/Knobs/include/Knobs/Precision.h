#ifndef CUTCP_KNOBS_PRECISION_H
#define CUTCP_KNOBS_PRECISION_H

#include <Vector/Vec3.h>

namespace Knobs 
{
    float GetCutoff(
        Vector::Vec3 minCoords,
        Vector::Vec3 maxCoords,
        float spacing,
        unsigned int precision
    );
}

#endif //CUTCP_KNOBS_PRECISION_H
