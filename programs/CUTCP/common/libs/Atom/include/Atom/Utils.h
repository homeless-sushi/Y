#ifndef CUTCP_ATOM_UTILS_H
#define CUTCP_ATOM_UTILS_H

#include "Atom/Atom.h"

#include <vector>

#include "Vector/Vec3.h"

namespace Atom
{
    void GetAtomBounds(
        const std::vector<Atom>& atoms,
        Vector::Vec3& min,
        Vector::Vec3& max
    );
}

#endif //CUTCP_ATOM_UTILS_H
