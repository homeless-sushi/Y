#ifndef BFS_BFS_BFS
#define BFS_BFS_BFS

#include <vector>

#include "Graph/Graph.h"

namespace Bfs
{
    class Bfs
    {
        public:
            Graph::Graph& graph;
            unsigned int source;
            int currentCost;           

            Bfs(Graph::Graph& graph, unsigned int source);
            virtual ~Bfs();
            virtual void run() = 0;
            virtual const std::vector<int>& getResult() = 0;
    };
 }

#endif //BFS_BFS_BFS
