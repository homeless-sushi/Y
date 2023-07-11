#ifndef BFS_GRAPH_READWRITE
#define BFS_GRAPH_READWRITE

#include "Graph/Graph.h"

#include <string>

namespace Graph 
{
    Graph ReadGraphFile(std::string fileURL);
    int WriteGraphFile(std::string fileURL);
}

#endif //BFS_GRAPH_READWRITE