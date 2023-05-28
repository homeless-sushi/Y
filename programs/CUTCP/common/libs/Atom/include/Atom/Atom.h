#ifndef CUTCP_ATOM_ATOM_H
#define CUTCP_ATOM_ATOM_H

#include <Vector/Vec3.h>

namespace Atom
{
    class Atom
    {
        public:
            Vector::Vec3 pos;
            float q;

            Atom() = default;
            Atom(float x, float y, float z, float q) :
                pos{x,y,z}, q{q}
            {};
            Atom(Vector::Vec3 pos, float q) :
                pos{pos}, q{q}
            {};
    };
}

#endif //CUTCP_ATOM_ATOM_H
