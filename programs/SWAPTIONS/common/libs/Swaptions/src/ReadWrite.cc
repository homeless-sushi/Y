#include <Swaptions/ReadWrite.h>
#include <Swaptions/Swaptions.h>

#include <iomanip>
#include <fstream>
#include <string>

namespace Swaptions 
{
    void WriteSwaptionsPrices(
        const std::string fileURL,
        unsigned int nSimulations,
        unsigned int nSwaptions,
        parm* swaptions
    )
    {
	    std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }
        
        outFile << std::setprecision(6);
        outFile << std::fixed;
        outFile << "#Swaptions: " << nSwaptions << "\n";
        outFile << "#Simulations: " << nSimulations << "\n";
        for (int i = 0; i < nSwaptions; i++){
            outFile << i << ": "
                << "[Price: " << swaptions[i].dSimSwaptionMeanPrice
                << " Error: " << swaptions[i].dSimSwaptionStdError << "]\n";
        }
    }
}
