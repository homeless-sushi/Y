#ifndef BFS_GRAPH_GRAPH
#define BFS_GRAPH_GRAPH

#include <vector>

namespace Graph 
{

    class Graph
    {
        public:
            unsigned int nVertices; // number of vertices
            unsigned int nEdges; // number of edges
            std::vector<unsigned int> edgeOffsets; //there are n_vertex+1 element, so that (edge_offset[i+1]-edge_offset[i]) is the degree of i
            std::vector<unsigned int> edges; //the vertexes adjacent to vertex i begin at edge_list[edge_offset[i]]

            Graph(unsigned int nVertices, unsigned int nEdges);

            Graph(const Graph& other);
            Graph& operator=(const Graph& other);

            Graph(Graph&& other);
            Graph& operator=(Graph&& other);
    };
}

#endif //BFS_GRAPH_GRAPH