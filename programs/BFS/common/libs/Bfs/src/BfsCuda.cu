#include "Bfs/Bfs.h"
#include "Bfs/BfsCuda.h"

#include <vector>

#include <cuda_runtime.h>

#include "Graph/Graph.h"

#include "CudaError/CudaError.h"

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
            CudaErrorCheck(
                cudaCreateTextureObject(dst, &resourceDesc, &texDesc, NULL)
            );
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

        CudaErrorCheck(
            cudaMalloc(&edgeOffsetsDevice_,sizeof(unsigned int)*graph.edgeOffsets.size())
        );
        CudaErrorCheck(
            cudaMalloc(&edgesDevice_, sizeof(unsigned int)*graph.edges.size())
        );
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(
            edgeOffsetsDevice_,
            graph.edgeOffsets.data(),
            sizeof(unsigned int)*graph.edgeOffsets.size(), 
            cudaMemcpyKind::cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            edgesDevice_,
            graph.edges.data(),
            sizeof(unsigned int)*graph.edges.size(),
            cudaMemcpyKind::cudaMemcpyHostToDevice
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataUploadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if(textureMemForEdgesOffsets_){
            memset(&edgeOffsetsTexture_, 0, sizeof(cudaTextureObject_t));
            createTextureObject(
                    edgeOffsetsDevice_,
                    &edgeOffsetsTexture_,
                    graph.edgeOffsets.size()
            );
        }
        if(textureMemForEdges_){
            memset(&edgesTexture_, 0, sizeof(cudaTextureObject_t));
            createTextureObject(
                edgesDevice_,
                &edgesTexture_,
                graph.edges.size()
            );
        }

        CudaErrorCheck(cudaMalloc(&costsDevice_, sizeof(int)*graph.nVertices));
        cudaMemset(costsDevice_, -1, sizeof(unsigned int)*graph.nVertices);
        cudaMemset(costsDevice_ + source, 0, sizeof(unsigned int));
        CudaErrorCheck(cudaMalloc(&doneDevice_, sizeof(bool)));
        cudaMemset(doneDevice_, true, sizeof(bool));
    }

    BfsCuda::~BfsCuda() 
    {   
        CudaErrorCheck(cudaFree(edgeOffsetsDevice_));
        CudaErrorCheck(cudaFree(edgesDevice_));

        if(textureMemForEdgesOffsets_){
            CudaErrorCheck(cudaDestroyTextureObject(edgeOffsetsTexture_));
        }
        if(textureMemForEdges_){
            CudaErrorCheck(cudaDestroyTextureObject(edgesTexture_));
        }

        CudaErrorCheck(cudaFree(costsDevice_));
        CudaErrorCheck(cudaFree(doneDevice_));
    }

    void BfsCuda::run()
    {
        const unsigned int blockSize = blockSize_;
        const unsigned int chunkSize = blockSize * chunkFactor_;
        const unsigned int gridSize = (graph.nVertices + blockSize - 1)/chunkSize + 1;

        bool done = false;
        while(!done){
            cudaMemset(doneDevice_, true, sizeof(bool));
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
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
            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            float kernelTime;
            cudaEventElapsedTime(&kernelTime, start, stop);
            kernelTotalTime+=kernelTime;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaMemcpy(
                &done,
                doneDevice_,
                sizeof(bool),
                cudaMemcpyKind::cudaMemcpyDeviceToHost
            );
            currentCost++;
        }
    }

    const std::vector<int>& BfsCuda::getResult() 
    {
        costsHost_.reserve(graph.nVertices);
        costsHost_.resize(costsHost_.capacity());
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(
            costsHost_.data(),
            costsDevice_,
            sizeof(int)*graph.nVertices,
            cudaMemcpyKind::cudaMemcpyDeviceToHost
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataDownloadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return costsHost_;
    };
}