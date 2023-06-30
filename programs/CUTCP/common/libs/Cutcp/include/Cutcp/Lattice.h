#ifndef CUTCP_CUTCP_LATTICE_H
#define CUTCP_CUTCP_LATTICE_H

#include <vector>

#include <cmath>

#include <Vector/Vec3.h>

#include <Atom/Atom.h>

namespace Lattice
{
    class Lattice
    {
        private:
            int nx_; // Number of lattice points in the X axis
            int ny_; // Number of lattice points in the Y axis
            int nz_; // Number of lattice points in the Z axis
            Vector::Vec3 min_; // Lowest corner of lattice
            float spacing_; // Spacing between potential points

        public:
            std::vector<float> points;

            Lattice() = default;
            Lattice(Vector::Vec3 minCoords, Vector::Vec3 maxCoords, float spacing) :
                min_{minCoords},
                spacing_{spacing},
                nx_{(int) floor((maxCoords.x-minCoords.x)/spacing) + 1},
                ny_{(int) floor((maxCoords.y-minCoords.y)/spacing) + 1},
                nz_{(int) floor((maxCoords.z-minCoords.z)/spacing) + 1},
                points(((nx_ * ny_ * nz_) + 7) & ~7, 0) // Round to multiple of 8
            {};

            int nx() const { return nx_; }
            int ny() const { return ny_; }
            int nz() const { return nz_; }
            const Vector::Vec3& min() const { return min_; }
            float spacing() const { return spacing_; }
    };
}

#endif //CUTCP_CUTCP_LATTICE_H

