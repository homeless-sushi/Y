#include "Graph/Graph.h"

namespace Graph 
{
    Graph::Graph(unsigned int nVertices, unsigned int nEdges) :
        nEdges(nEdges),
        nVertices(nVertices) 
    {
        edgeOffsets.reserve(nVertices+1);
        edges.reserve(nEdges);
    }

    Graph::Graph(const Graph& other) : 
        nVertices{other.nVertices},
        nEdges{other.nEdges},
        edgeOffsets(other.edgeOffsets),
        edges(other.edges){}
    
    Graph& Graph::operator=(const Graph& other)
    {
        nVertices = other.nVertices;
        nEdges = other.nEdges;
        edgeOffsets = other.edgeOffsets;
        edges = other.edges;

        return *this;
    }

    Graph::Graph(Graph&& other) :
        nVertices{other.nVertices},
        nEdges{other.nEdges},
        edgeOffsets{std::move(other.edgeOffsets)},
        edges{std::move(other.edges)}{}

    Graph& Graph::operator=(Graph&& other)
    {
        nVertices = other.nVertices;
        nEdges = other.nEdges;
        edgeOffsets = std::move(other.edgeOffsets);
        edges = std::move(other.edges);

        return *this;
    }
}