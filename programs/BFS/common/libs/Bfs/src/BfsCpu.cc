#include "Bfs/Bfs.h"
#include "Bfs/BfsCpu.h"

#include <iostream>

#include <omp.h>

#include "Graph/Graph.h"

namespace Bfs 
{
    BfsCpu::BfsCpu(Graph::Graph& graph, unsigned int source, unsigned int nThreads) :
        Bfs(graph, source),
        nThreads_(nThreads),
        costs_(graph.nVertices, -1)
    {
        costs_[source] = 0;
    }
    BfsCpu::~BfsCpu() = default;

    const std::vector<int>& BfsCpu::getResult() { return costs_; }

    bool BfsCpu::run() 
    {
        bool done = true;
        #pragma omp parallel for \
        num_threads(nThreads_)
        for(int fromNode = 0; fromNode < graph.nVertices; fromNode++){
            if (costs_[fromNode] == currentCost){
                const int nodeEdgesStart = graph.edgeOffsets[fromNode];
                const int nodeEdgesStop = graph.edgeOffsets[fromNode+1];
                for(int edgeId = nodeEdgesStart; edgeId < nodeEdgesStop; edgeId++){
                    const int toNode = graph.edges[edgeId];
                    if(costs_[toNode] == -1){
                        costs_[toNode]=currentCost+1;
                        done=false;
                    }
                }
            }
        }

        currentCost++;
        return done;
    }
}
