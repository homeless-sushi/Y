#include "Histo/ReadWrite.h"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "Histo/Const.h"

namespace Histo
{
    void ReadDataFile(
        const std::string& fileURL,
        unsigned& imgWidth, unsigned& imgHeight,
        std::vector<unsigned short>& rgb
    )
    {       
        std::ifstream inFile(fileURL);
        if (!inFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        std::string lineString;
        getline(inFile, lineString);
        std::istringstream lineStream(lineString);
        lineStream >> imgWidth;
        lineStream >> imgHeight;
        const unsigned n = imgWidth*imgHeight*N_CHANNELS;

        rgb.reserve(n);
        while(inFile.good()){

            getline(inFile, lineString);
            lineStream = std::istringstream(lineString);

            if(lineString.empty())
                break;

            while(lineStream.good()){
                unsigned short data;
                lineStream >> data;
                rgb.push_back(data);
            }
        }
    };

    void ReadBinaryDataFile(
        const std::string& fileURL,
        unsigned& imgWidth, unsigned& imgHeight,
        std::vector<unsigned short>& rgb
    )
    {
        std::ifstream inFile(fileURL, std::ios::binary);
        if (!inFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        inFile.read(reinterpret_cast<char*>(&imgWidth), sizeof(unsigned));
        inFile.read(reinterpret_cast<char*>(&imgHeight), sizeof(unsigned));

        const unsigned n = imgWidth * imgHeight * N_CHANNELS;

        rgb.resize(n);
        inFile.read(reinterpret_cast<char*>(rgb.data()), n * sizeof(unsigned short));
    };
        
    void WriteHistogramFile(
        const std::string fileURL,
        const std::vector<unsigned>& histo
    )
    {
        std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        unsigned n = N_CHANNELS * N_CHANNEL_VALUES;
        unsigned i = 0;
        outFile << "RED" << "\n";
        for(; i < 1*N_CHANNEL_VALUES; ++i)
            outFile << histo[i] << " ";
        outFile << "\n";

        outFile << "GREEN" << "\n"; 
        for(; i < 2*N_CHANNEL_VALUES; ++i)
            outFile << histo[i] << " ";
        outFile << "\n";

        outFile << "BLUE" << "\n"; 
        for(; i < 3*N_CHANNEL_VALUES; ++i)
            outFile << histo[i] << " ";
        outFile << std::endl;
    };
}