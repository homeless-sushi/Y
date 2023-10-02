#include <Dummy/Dummy.h>

#include <vector>

#include <cuda_runtime.h>

#include "CudaError/CudaError.h"

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
        CudaErrorCheck(cudaMalloc(&gpu_in, sizeof(float)*data.size()));
        CudaErrorCheck(cudaMalloc(&gpu_out, sizeof(float)*data.size()));
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(
            gpu_in,
            data.data(),
            sizeof(float)*data.size(),
            cudaMemcpyHostToDevice
        );
        cudaMemset(gpu_out, 0, sizeof(float)*data.size());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataUploadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    Dummy::~Dummy()
    {
        CudaErrorCheck(cudaFree(gpu_in));
        CudaErrorCheck(cudaFree(gpu_out));
    };

    std::vector<float> Dummy::getResult()
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(
            data.data(),
            gpu_out,
            sizeof(float)*data.size(),
            cudaMemcpyDeviceToHost
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataDownloadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
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

    __global__
    void dummyInfinteKernel(float* in, float* out, unsigned int n)
    {
        unsigned int absolute_idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = gridDim.x * blockDim.x;

        while(true){
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
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        dummyKernel<<<gridLen, blockLen>>>(gpu_in, gpu_out, data.size(), times);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&kernelTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    void Dummy::runInfinite()
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        dummyInfinteKernel<<<gridLen, blockLen>>>(gpu_in, gpu_out, data.size());
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&kernelTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };
}
