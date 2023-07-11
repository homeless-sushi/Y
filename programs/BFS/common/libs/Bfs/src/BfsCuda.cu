#include "Bfs/Bfs.h"
#include "Bfs/BfsCuda.h"

#include <vector>

#include <cuda_runtime.h>

#include "Graph/Graph.h"

namespace Bfs
{
    namespace
    {
        void createTextureObject(unsigned int* src, cudaTextureObject_t* dst, size_t size)
        {
            cudaResourceDesc resourceDesc;
            memset(&resourceDesc, 0, sizeof(resourceDesc));
            resourceDesc.resType = cudaResourceTypeLinear;
            resourceDesc.res.linear.devPtr = src;
            resourceDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
            resourceDesc.res.linear.desc.x = sizeof(unsigned int)*CHAR_BIT;
            resourceDesc.res.linear.sizeInBytes = sizeof(unsigned int)*size;
            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.readMode = cudaReadModeElementType;
            cudaCreateTextureObject(dst, &resourceDesc, &texDesc, NULL);
        }

        __global__
        void kernel(
            unsigned int nVertices,
            unsigned int chunkSize,
            unsigned int *edgeOffsetsDevice,
            unsigned int *edgesDevice,
            cudaTextureObject_t edgeOffsetsTexture,
            cudaTextureObject_t edgesTexture,
            bool textureMemForEdgesOffsets,
            bool textureMemForEdges,
            int *costs,
            int currCost,
            bool *done)
        {
            const unsigned int startNode = blockIdx.x*chunkSize;
            const unsigned int stopNode = min(startNode + chunkSize, nVertices);
            for(unsigned int fromNode = startNode; fromNode < stopNode; fromNode++) {
                if(costs[fromNode] == currCost) {
                    unsigned int nodeEdgesStart;
                    unsigned int nodeEdgesEnd;
                    if(textureMemForEdgesOffsets){
                        nodeEdgesStart = tex1Dfetch<unsigned int>(edgeOffsetsTexture, fromNode);
                        nodeEdgesEnd = tex1Dfetch<unsigned int>(edgeOffsetsTexture, fromNode+1);
                    }else{
                        nodeEdgesStart = edgeOffsetsDevice[fromNode];
                        nodeEdgesEnd = edgeOffsetsDevice[fromNode+1];
                    }
                    
                    for(unsigned int i = nodeEdgesStart + threadIdx.x; i < nodeEdgesEnd; i+=blockDim.x) {
                        unsigned int toNode;

                        if(textureMemForEdges){
                            toNode = tex1Dfetch<unsigned int>(edgesTexture, i);
                        }else{
                            toNode = edgesDevice[i];
                        }

                        if(costs[toNode] == -1) {
                            costs[toNode] = currCost + 1;
                            *done = false;
                        }
                    }
                }
            }
        }
    }

    BfsCuda::BfsCuda(    
        Graph::Graph& graph,
        unsigned int source,
        unsigned int blockSize,
        unsigned int chunkFactor, 
        bool textureMemForEdgesOffsets,
        bool textureMemForEdges
    ) :
        Bfs(graph, source),
        blockSize_(blockSize),
        chunkFactor_(chunkFactor),
        textureMemForEdgesOffsets_(textureMemForEdgesOffsets),
        textureMemForEdges_(textureMemForEdges)
    {
        cudaMalloc(&edgeOffsetsDevice_, sizeof(unsigned int)*graph.edgeOffsets.size());
        cudaMemcpy(edgeOffsetsDevice_, graph.edgeOffsets.data(), sizeof(unsigned int)*graph.edgeOffsets.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        if(textureMemForEdgesOffsets_){
            memset(&edgeOffsetsTexture_, 0, sizeof(cudaTextureObject_t));
            createTextureObject(edgeOffsetsDevice_, &edgeOffsetsTexture_, graph.edgeOffsets.size());
        }

        cudaMalloc(&edgesDevice_, sizeof(unsigned int)*graph.edges.size());
        cudaMemcpy(edgesDevice_, graph.edges.data(), sizeof(unsigned int)*graph.edges.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        if(textureMemForEdges_){
            memset(&edgesTexture_, 0, sizeof(cudaTextureObject_t));
            createTextureObject(edgesDevice_, &edgesTexture_, graph.edges.size());
        }

        cudaMalloc(&costsDevice_, sizeof(int)*graph.nVertices);
        cudaMemset(costsDevice_, -1, sizeof(unsigned int)*graph.nVertices);
        cudaMemset(costsDevice_ + source, 0, sizeof(unsigned int));

        cudaMalloc(&doneDevice_, sizeof(bool));
        cudaMemset(doneDevice_, true, sizeof(bool));
    }

    BfsCuda::~BfsCuda() 
    {
        cudaFree(edgeOffsetsDevice_);
        cudaFree(edgesDevice_);

        if(textureMemForEdgesOffsets_){
            cudaDestroyTextureObject(edgeOffsetsTexture_);
        }
        if(textureMemForEdges_){
            cudaDestroyTextureObject(edgesTexture_);
        }

        cudaFree(costsDevice_);
        cudaFree(doneDevice_);
    }

    bool BfsCuda::run()
    {
        cudaMemset(doneDevice_, true, sizeof(bool));
            

        const unsigned int blockSize = blockSize_;
        const unsigned int chunkSize = blockSize * chunkFactor_;
        const unsigned int gridSize = (graph.nVertices + blockSize - 1)/chunkSize + 1;

        kernel<<<gridSize, blockSize>>>(
            graph.nVertices, 
            chunkSize,
            edgeOffsetsDevice_, 
            edgesDevice_,
            edgeOffsetsTexture_,
            edgesTexture_,
            textureMemForEdgesOffsets_,
            textureMemForEdges_,
            costsDevice_,
            currentCost,
            doneDevice_
        );
        
        currentCost++;
        bool done;
        cudaMemcpy(&done, doneDevice_, sizeof(bool), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        return done;
    }

    const std::vector<int>& BfsCuda::getResult() 
    {
        costsHost_.reserve(graph.nVertices);
        costsHost_.resize(costsHost_.capacity());
        cudaMemcpy(costsHost_.data(), costsDevice_, sizeof(int)*graph.nVertices, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        return costsHost_;
    };
}