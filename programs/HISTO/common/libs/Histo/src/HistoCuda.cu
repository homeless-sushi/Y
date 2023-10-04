
#include <Histo/Const.h>
#include <Histo/Histo.h>
#include <Histo/HistoCuda.h>

#include <vector>

#include <stdint.h>

namespace Histo 
{
    HistoCuda::HistoCuda(
        std::vector<unsigned short> rgb,
        unsigned blockSize,
        unsigned nBlocks
    ) :
        ::Histo::Histo{rgb},
        blockSize(blockSize),
        nBlocks(nBlocks)
    {
        cudaMalloc(&rgb_device, rgb.size()*sizeof(unsigned short));
        cudaMalloc(&histo_partial_device, histo.size()*nBlocks*sizeof(unsigned));
        cudaMalloc(&histo_final_device, histo.size()*sizeof(unsigned));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(rgb_device, rgb.data(), rgb.size()*sizeof(unsigned short), cudaMemcpyHostToDevice);
        cudaMemset(histo_partial_device, 0, histo.size()*nBlocks*sizeof(unsigned));
        cudaMemset(histo_final_device, 0, histo.size()*sizeof(unsigned));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataUploadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    __global__
    void histogram_partial(
        const unsigned short *in,
        unsigned n,
        unsigned *out
    )
    {
        const int absoluteThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        const unsigned nPixels = n/3;

        // initialize temporary accumulation array in shared memory
        __shared__ unsigned int smem[N_CHANNELS * N_CHANNEL_VALUES];
        for (int i = threadIdx.x; i < N_CHANNELS * N_CHANNEL_VALUES; i += blockDim.x) 
            smem[i] = 0;
        __syncthreads();

        // process pixels
        // updates our block's partial histogram in shared memory
        for (int i = absoluteThreadIdx; i < nPixels; i+=stride){
            const unsigned short r = in[i*N_CHANNELS];
            const unsigned short g = in[i*N_CHANNELS+1];
            const unsigned short b = in[i*N_CHANNELS+2];
            atomicAdd(&smem[N_CHANNEL_VALUES*0 + r], 1);
            atomicAdd(&smem[N_CHANNEL_VALUES*1 + g], 1);
            atomicAdd(&smem[N_CHANNEL_VALUES*2 + b], 1);
        }
        __syncthreads();

        // write partial histogram into the global memory
        unsigned* const block_out = out + N_CHANNELS * N_CHANNEL_VALUES * blockIdx.x;
        for (int i = threadIdx.x; i < N_CHANNEL_VALUES; i += blockDim.x) {
            block_out[i + N_CHANNEL_VALUES * 0] = smem[i + N_CHANNEL_VALUES * 0];
            block_out[i + N_CHANNEL_VALUES * 1] = smem[i + N_CHANNEL_VALUES * 1];
            block_out[i + N_CHANNEL_VALUES * 2] = smem[i + N_CHANNEL_VALUES * 2];
        }
    }

    __global__
    void histogram_final(
        const unsigned int *in,
        int nBlocks,
        unsigned int *out
    )
    {
        const int absoluteThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (absoluteThreadIdx < N_CHANNELS * N_CHANNEL_VALUES) {
            unsigned total = 0;
            for (unsigned currBlock = 0; currBlock < nBlocks; currBlock++){
                unsigned offset = currBlock * N_CHANNELS * N_CHANNEL_VALUES;
                total += in[absoluteThreadIdx + offset];
            }
            out[absoluteThreadIdx] = total;
        }
    }

    void HistoCuda::run()
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        histogram_partial<<<nBlocks, blockSize>>>(rgb_device, rgb.size(), histo_partial_device);
        unsigned nBlocksFinal = (N_CHANNELS*N_CHANNEL_VALUES + blockSize - 1)/blockSize;
        histogram_final<<<nBlocksFinal, blockSize>>>(histo_partial_device, nBlocks, histo_final_device);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&kernelTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    std::vector<unsigned> HistoCuda::getResult()
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(histo.data(), histo_final_device, histo.size()*sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataDownloadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return histo;
    };

    HistoCuda::~HistoCuda()
    {
        cudaFree(rgb_device);
        cudaFree(histo_partial_device);
        cudaFree(histo_final_device);
    };
}
