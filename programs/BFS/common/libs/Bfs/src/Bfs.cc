#include "Bfs/Bfs.h"

#include "Graph/Graph.h"

namespace Bfs 
{
    Bfs::Bfs(Graph::Graph& graph, unsigned int source) :
        graph(graph),
        source{source},
        currentCost{0}
    {}
    Bfs::~Bfs() = default;
}
