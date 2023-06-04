#include <Dummy/Dummy.h>

#include <vector>

#include <cuda_runtime.h>

namespace Dummy 
{
    
    Dummy::Dummy(
        std::vector<float> data,
        unsigned int gridLen,
        unsigned int blockLen,
        unsigned int times
    ) : 
        data{data},
        gridLen{gridLen},
        blockLen{blockLen},
        times{times}
    {
        cudaMalloc(&gpu_in, sizeof(float)*data.size());
        cudaMemcpy(gpu_in, data.data(), sizeof(float)*data.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&gpu_out, sizeof(float)*data.size());
        cudaMemset(gpu_out, 0, sizeof(float)*data.size());
    };

    Dummy::~Dummy()
    {
        cudaFree(gpu_in);
        cudaFree(gpu_out);
    };

    std::vector<float> Dummy::getResult()
    {
        cudaMemcpy(data.data(), gpu_out, sizeof(float)*data.size(), cudaMemcpyDeviceToHost);
        return data;
    }

    __global__
    void dummyKernel(float* in, float* out, unsigned int n, unsigned int times)
    {
        unsigned int absolute_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = gridDim.x * blockDim.x;

        for(unsigned int t = 0; t < times; t++){
            for(int i = absolute_idx; i < n; i+=stride){
                out[i]=in[i] * i;
            }
            for(int i = absolute_idx; i < n; i+=stride){
                out[i]=in[i] / i;
            }
        }
    };

    void Dummy::run()
    {
        dummyKernel<<<gridLen, blockLen>>>(gpu_in, gpu_out, data.size(), times);
        cudaDeviceSynchronize();
    };
}
