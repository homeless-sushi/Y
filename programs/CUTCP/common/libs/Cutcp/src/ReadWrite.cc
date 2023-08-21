#include <Cutcp/Lattice.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

namespace Cutcp
{
    void WriteLattice(
        const std::string fileURL,
        const Lattice::Lattice& lattice)
    {
	    std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }
        
        outFile << std::fixed << std::setprecision(6);
        for (const auto& potential : lattice.points){
            outFile << potential << "\n";
        }
        std::cout << std::endl;
    }
}
