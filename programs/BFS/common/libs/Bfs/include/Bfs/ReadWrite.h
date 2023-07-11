#ifndef BFS_BFS_READWRITE
#define BFS_BFS_READWRITE

#include "Bfs/Bfs.h"

#include <string>
#include <vector>

#include "Graph/Graph.h"

namespace Bfs
{
    int WriteCosts(std::string fileURL, std::vector<int>& costs);
}

#endif //BFS_BFS_READWRITE
