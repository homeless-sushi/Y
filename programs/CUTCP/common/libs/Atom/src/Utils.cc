#include "Atom/Atom.h"
#include "Atom/Utils.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Vector/Vec3.h>

namespace Atom
{
    void GetAtomBounds(
        const std::vector<Atom>& atoms,
        Vector::Vec3& min,
        Vector::Vec3& max
    )
    {
        if(!atoms.size()){
            throw std::runtime_error("Empty atoms vector");
        }
        
        max = min = atoms.front().pos;

        for(const auto& atom : atoms) {
            min.x = std::min(min.x, atom.pos.x);
            max.x = std::max(max.x, atom.pos.x);
            min.y = std::min(min.y, atom.pos.y);
            max.y = std::max(max.y, atom.pos.y);
            min.z = std::min(min.z, atom.pos.z);
            max.z = std::max(max.z, atom.pos.z);
        }
    }
}
