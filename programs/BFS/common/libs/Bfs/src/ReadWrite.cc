#include "Bfs/Bfs.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "Graph/Graph.h"

namespace Bfs
{
    int WriteCosts(std::string fileURL, std::vector<int>& costs)
    {
	    std::ofstream resultFile (fileURL);
        if (!resultFile.is_open()){
            std::cerr << "Error: Cannot open " << fileURL << std::endl;
            return -1;
        }

        for (int cost : costs){
            resultFile << cost << std::endl;
        }

        return 0;
    }
}
