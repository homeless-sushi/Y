#ifndef CUTCP_CUTCP_CUTCP_H
#define CUTCP_CUTCP_CUTCP_H

#include <Cutcp/Lattice.h>

#include <vector>

#include <Atom/Atom.h>

#define CELL_LEN      4.f
#define INV_CELL_LEN  (1.f/CELL_LEN)

namespace Cutcp
{
    class Cutcp
    {
        public: 
            virtual ~Cutcp() = default;

            virtual void run() = 0;
            virtual Lattice::Lattice getResult() = 0;            
    };
}

#endif //CUTCP_CUTCP_CUTCP_H
