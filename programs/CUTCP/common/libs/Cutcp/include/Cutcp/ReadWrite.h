#ifndef CUTCP_CUTCP_READWRITE_H
#define CUTCP_CUTCP_READWRITE_H

#include <Cutcp/Lattice.h>

#include <string>

namespace Cutcp
{
    void WriteLattice(
        const std::string fileURL,
        const Lattice::Lattice& lattice
    );
}

#endif //CUTCP_CUTCP_UTILS_H
