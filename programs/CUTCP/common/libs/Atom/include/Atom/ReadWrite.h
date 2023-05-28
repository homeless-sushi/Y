#ifndef CUTCP_ATOM_READWRITE_H
#define CUTCP_ATOM_READWRITE_H

#include "Atom/Atom.h"

#include <vector>
#include <string>

namespace Atom
{
    std::vector<Atom> ReadAtomFile(std::string fileURL);
    void WriteAtomFile(
        const std::string fileURL,
        const std::vector<Atom>& atoms);
}

#endif //CUTCP_ATOM_READWRITE_H
