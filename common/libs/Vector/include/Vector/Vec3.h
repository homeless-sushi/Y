#ifndef COMMON_VECTOR_VECTOR3_H
#define COMMON_VECTOR_VECTOR3_H

#include <cmath>

namespace Vector
{
    class Vec3
    {
        public:
            float x;
            float y;
            float z;

            Vec3() = default;
            Vec3(float u) :
                x{u}, y{u}, z{u}
            {};
            Vec3(float x, float y, float z) :
                x{x}, y{y}, z{z}
            {};

            Vec3 operator+(const Vec3& other){ return Vec3(x+other.x, y+other.y, z+other.z); };
            Vec3 operator-(const Vec3& other){ return Vec3(x-other.x, y-other.y, z-other.z); };
            Vec3 operator-(){ return Vec3(-x, -y, -z); };
            Vec3 operator*(float a){ return Vec3(a*x, a*y, a*z); };

            float distance(const Vec3& other){
                float dx = (x-other.x);
                float dy = (y-other.y);
                float dz = (z-other.z);
                return sqrt(dx*dx+dy*dy+dz*dz);
            }
    };
}

#endif //COMMON_VECTOR_VECTOR3_H
