#ifndef BFS_KNOBS_DEVICE
#define BFS_KNOBS_DEVICE

#include <string>

namespace Knobs
{
    enum DEVICE
    {
        CPU,
        GPU
    };

    std::string DeviceToString(DEVICE device);

    struct DeviceKnobs
    {   
        public:
            const DEVICE device;
        
        protected:
            DeviceKnobs(DEVICE device);
    };

    class CpuKnobs : public DeviceKnobs
    {   
        public:
            unsigned int N_THREADS;

            CpuKnobs(unsigned int N_THREADS);
    };

    class GpuKnobs : public DeviceKnobs
    {   
        public:
            enum BLOCK_SIZE
            {
                BLOCK_32 = 32,
                BLOCK_64 = 64,
                BLOCK_128 = 128,
                BLOCK_256 = 256,
                BLOCK_512 = 512,
                BLOCK_1024 = 1024
            };
            BLOCK_SIZE blockSize;

            enum CHUNK_FACTOR
            {
                CHUNK_1 = 1,
                CHUNK_2 = 2,
                CHUNK_4 = 4,
                CHUNK_8 = 8,
            };
            CHUNK_FACTOR chunkSize;

            enum MEMORY_TYPE
            {
                DEVICE_MEM = false,
                TEXTURE_MEM = true

            };
            MEMORY_TYPE edgesOffsets;
            MEMORY_TYPE edges;

            GpuKnobs(
                BLOCK_SIZE BLOCK_SIZE,
                CHUNK_FACTOR CHUNK_FACTOR,
                MEMORY_TYPE EDGES_OFFSETS_MEM_TYPE,
                MEMORY_TYPE EDGES_MEM_TYPE
            );
    };
}

#endif //BFS_KNOBS_DEVICE
