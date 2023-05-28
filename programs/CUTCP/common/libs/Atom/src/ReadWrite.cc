#include "Atom/Atom.h"
#include "Atom/ReadWrite.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <iostream>

namespace Atom
{
    std::vector<Atom> ReadAtomFile(const std::string fileURL)
    {
        std::ifstream atomFile(fileURL);
        if (!atomFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        std::vector<Atom> atoms;
        while(atomFile.good()){

            std::string lineString;
            getline(atomFile, lineString);

            if(lineString.empty())
                break;

            std::istringstream lineStream(lineString);
            Atom atom;
            lineStream >> atom.pos.x;
            lineStream >> atom.pos.y;
            lineStream >> atom.pos.z;
            lineStream >> atom.q;
            atoms.push_back(atom);
        }
        
        return atoms;
    }

    void WriteAtomFile(
        const std::string fileURL,
        const std::vector<Atom>& atoms)
    {
	    std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        outFile << std::fixed << std::setprecision(6);
        for (const auto& atom : atoms){
            outFile << 
                atom.pos.x << " " << 
                atom.pos.y << " " << 
                atom.pos.z << " " << 
                atom.q << std::endl;
        }
    }
}
