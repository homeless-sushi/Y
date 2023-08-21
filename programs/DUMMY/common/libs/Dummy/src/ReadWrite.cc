#include "Dummy/ReadWrite.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Dummy
{
    void ReadFile(
        const std::string& fileURL,
        std::vector<float>& data
    )
    {
        std::ifstream file(fileURL);
        if (!file.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        std::string lineString;
        while(file.good()){

            getline(file, lineString);

            if(lineString.empty())
                break;

            std::istringstream lineStream(lineString);
            float datapoint;
            lineStream >> datapoint;

            data.push_back(datapoint);
        }
    }

    void WriteFile(
        const std::string fileURL,
        const std::vector<float>& data
    )
    {
	    std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        for (const float datapoint : data){
            outFile << datapoint << "\n";
        }
        outFile << std::endl;
    }
}
