#include <Knobs/Precision.h>

#include <cmath>

#include <Vector/Vec3.h>

namespace Knobs 
{
    float GetCutoff(
        Vector::Vec3 minCoords,
        Vector::Vec3 maxCoords,
        float spacing,
        unsigned int precision
    ){
        float dist = minCoords.distance(maxCoords);
        return dist*precision/100.f;
    };
}
