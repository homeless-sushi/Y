#ifndef NBODY_NBODY_BODY_H
#define NBODY_NBODY_BODY_H

#include "Vector/Vec3.h"

namespace Nbody 
{
    class Body {

        public:
            Vector::Vec3 pos;
            Vector::Vec3 vel;

            Body() = default;
            Body(Vector::Vec3 pos, Vector::Vec3 vel) :
                pos{pos}, vel{vel}
            {};
            Body(
                float x, float y, float z, 
                float vx, float vy, float vz) :
                pos{x,y,z}, vel{vx,vy,vz}
            {};
    };
}

#endif //NBODY_NBODY_BODY_H
