#ifndef BFS_BFS_CUDA
#define BFS_BFS_CUDA

#include "Bfs/Bfs.h"

#include <vector>

#include "cuda_runtime_api.h"

#include "Graph/Graph.h"

namespace Bfs
{
    class BfsCuda : public ::Bfs::Bfs
    {
        public:
            BfsCuda(
                Graph::Graph& graph,
                unsigned int source,
                unsigned int blockSize,
                unsigned int chunkFactor, 
                bool textureMemForEdgesOffsets,
                bool textureMemForEdges
            );
            virtual ~BfsCuda() override;

            virtual bool run() override;
            virtual const std::vector<int>& getResult() override;

        private:
            unsigned int blockSize_;
            unsigned int chunkFactor_; 
            bool textureMemForEdgesOffsets_;
            bool textureMemForEdges_;

            unsigned int* edgeOffsetsDevice_;
            unsigned int* edgesDevice_;
            cudaTextureObject_t edgeOffsetsTexture_;
            cudaTextureObject_t edgesTexture_;
            int* costsDevice_;
            bool* doneDevice_;

            std::vector<int> costsHost_;
    };
}

#endif //BFS_BFS_CUDA