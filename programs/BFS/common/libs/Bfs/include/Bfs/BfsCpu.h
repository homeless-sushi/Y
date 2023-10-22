#ifndef BFS_BFS_CPU
#define BFS_BFS_CPU

#include "Bfs/Bfs.h"

#include <vector>

#include "Graph/Graph.h"

namespace Bfs
{
    class BfsCpu : public ::Bfs::Bfs
    {
        public:
            BfsCpu(Graph::Graph& graph, unsigned int source, unsigned int nThreads);
            virtual ~BfsCpu() override;

            virtual void run() override;
            virtual const std::vector<int>& getResult() override;

        private:
            unsigned int nThreads_;
            
            std::vector<int> costs_;
    };
}

#endif //BFS_BFS_CPU
