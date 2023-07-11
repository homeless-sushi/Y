#include "Graph/Graph.h"
#include "Graph/ReadWrite.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>

namespace Graph 
{
    Graph ReadGraphFile(std::string fileURL)
    {
	    std::ifstream graphFile(fileURL);
        if (!graphFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }
        
        std::string lineString;
        getline(graphFile, lineString);
        std::istringstream firstLineStream(lineString);
        unsigned int nVertices;
        firstLineStream >> nVertices;
        unsigned int nEdges;
        firstLineStream >> nEdges;
        Graph graph(nVertices, nEdges);
        
        unsigned int nEdgesRead = 0;
        graph.edgeOffsets.push_back(nEdgesRead);
        while(graphFile.good()){

            getline(graphFile, lineString);
            std::istringstream lineStream(lineString);
            while(lineStream.good()){
                unsigned int destinationVertex;
                lineStream >> destinationVertex;
                graph.edges.push_back(destinationVertex);
                ++nEdgesRead;
            }

            graph.edgeOffsets.push_back(nEdgesRead);
        }
        graphFile.close();
        return graph;
    }

    int WriteGraphFile(std::string fileURL, Graph& graph)
    {
	    std::ofstream graphFile (fileURL);
        if (!graphFile.is_open()){
            std::cerr << "Error: Cannot open " << fileURL << std::endl;
            return -1;
        }
        
        graphFile << graph.nVertices << " " << graph.nEdges << std::endl;
        
        for(unsigned int nodeId = 0; nodeId < graph.nVertices + 1; ++nodeId){
            const unsigned int startOffset = graph.edgeOffsets[nodeId];
            const unsigned int endOffset = graph.edgeOffsets[nodeId+1];
            for(unsigned int edgeId = startOffset; edgeId < endOffset; ++edgeId){
                graphFile << " " << graph.edges[edgeId];
            }
            graphFile << std::endl;
        }
        graphFile << std::endl;

        return 0;
    }
}