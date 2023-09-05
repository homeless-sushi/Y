#include "Nbody/Body.h"
#include "Nbody/Nbody.h"
#include "Nbody/NbodyCuda.h"

#include <vector>

#include <cuda_runtime.h>

#include "CudaError/CudaError.h"

namespace NbodyCuda 
{
    BodySoa::BodySoa() :
        owner{true},
        x{nullptr},
        y{nullptr},
        z{nullptr},
        vx{nullptr},
        vy{nullptr},
        vz{nullptr},
        n{0}
    {};

    BodySoa::BodySoa(unsigned long n) :
        owner{true},
        n{n}
    {
        CudaErrorCheck(cudaMalloc(&x, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&y, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&z, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&vx, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&vy, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&vz, sizeof(float)*n));
    };

    BodySoa::BodySoa(const std::vector<::Nbody::Body>& bodies) :
        owner{true},
        n{bodies.size()}
    {
        float* const tmp_x = (float*) malloc(sizeof(float)*n);
        float* const tmp_y = (float*) malloc(sizeof(float)*n);
        float* const tmp_z = (float*) malloc(sizeof(float)*n);
        float* const tmp_vx = (float*) malloc(sizeof(float)*n);
        float* const tmp_vy = (float*) malloc(sizeof(float)*n);
        float* const tmp_vz = (float*) malloc(sizeof(float)*n);
        for(unsigned long i = 0; i < n; ++i){
            tmp_x[i] = bodies[i].pos.x;
            tmp_y[i] = bodies[i].pos.y;
            tmp_z[i] = bodies[i].pos.z;
            tmp_vx[i] = bodies[i].vel.x;
            tmp_vy[i] = bodies[i].vel.y;
            tmp_vz[i] = bodies[i].vel.z;
        }
        CudaErrorCheck(cudaMalloc(&x, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&y, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&z, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&vx, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&vy, sizeof(float)*n));
        CudaErrorCheck(cudaMalloc(&vz, sizeof(float)*n));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        CudaErrorCheck(
            cudaMemcpy(x, tmp_x, sizeof(float)*n, cudaMemcpyHostToDevice)
        );
        CudaErrorCheck(
            cudaMemcpy(y, tmp_y, sizeof(float)*n, cudaMemcpyHostToDevice)
        );
        CudaErrorCheck(
            cudaMemcpy(z, tmp_z, sizeof(float)*n, cudaMemcpyHostToDevice)
        );
        CudaErrorCheck(
            cudaMemcpy(vx, tmp_vx, sizeof(float)*n, cudaMemcpyHostToDevice)
        );
        CudaErrorCheck(
            cudaMemcpy(vy, tmp_vy, sizeof(float)*n, cudaMemcpyHostToDevice)
        );
        CudaErrorCheck(
            cudaMemcpy(vz, tmp_vz, sizeof(float)*n, cudaMemcpyHostToDevice)
        );
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&dataUploadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        free(tmp_x);
        free(tmp_y);
        free(tmp_z);
        free(tmp_vx);
        free(tmp_vy);
        free(tmp_vz);
    }

    BodySoa::BodySoa(const BodySoa& owner) :
        owner{false},
        x{owner.x},
        y{owner.y},
        z{owner.z},
        vx{owner.vx},
        vy{owner.vy},
        vz{owner.vz}
    {};

    BodySoa::~BodySoa()
    {   
        n = 0;
        if(owner){
            cudaFree(x);
            cudaFree(y);
            cudaFree(z);
            cudaFree(vx);
            cudaFree(vy);
            cudaFree(vz);
        }
        x = nullptr;
        y = nullptr;
        z = nullptr;
        vx = nullptr;
        vy = nullptr;
        vz = nullptr;
    }

    std::vector<::Nbody::Body> BodySoa::getBodiesVector()
    {
        float* const tmp_x = (float*) malloc(sizeof(float)*n);
        float* const tmp_y = (float*) malloc(sizeof(float)*n);
        float* const tmp_z = (float*) malloc(sizeof(float)*n);
        float* const tmp_vx = (float*) malloc(sizeof(float)*n);
        float* const tmp_vy = (float*) malloc(sizeof(float)*n);
        float* const tmp_vz = (float*) malloc(sizeof(float)*n);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(tmp_x, x, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp_y, y, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp_z, z, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp_vx, vx, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp_vy, vy, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp_vz, vz, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&dataDownloadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
   
        std::vector<::Nbody::Body> bodies;
        bodies.reserve(n);
        for(unsigned long i = 0; i < n; ++i){
            ::Nbody::Body body(
                tmp_x[i],
                tmp_y[i],
                tmp_z[i],
                tmp_vx[i],
                tmp_vy[i],
                tmp_vz[i]
            );
            bodies.push_back(body);
        }
        return bodies;
    }

    NbodyCuda::NbodyCuda(
        const std::vector<::Nbody::Body>& bodies,
        float simulationTime,
        float timeStep,
        unsigned int blockSize
    ) :
        Nbody::Nbody(simulationTime, timeStep),
        n{bodies.size()},
        blockSize{blockSize},
        in{bodies},
        out{n},
        dataUploadTime{in.getDataUploadTime()}
    {};

    NbodyCuda::~NbodyCuda() = default;

    __global__
    void kernel(
        float* in_x, float* in_y, float* in_z, float* in_vx, float* in_vy, float* in_vz,
        float* out_x, float* out_y, float* out_z, float* out_vx, float* out_vy, float* out_vz, 
        float dt, int n) 
    {
        const int absoluteThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        extern __shared__ float curr_block_xyz[];
        float* curr_block_x = curr_block_xyz;
        float* curr_block_y = curr_block_xyz + blockDim.x;
        float* curr_block_z = curr_block_xyz + 2 * blockDim.x;

        for(int i = absoluteThreadIdx; i < n; i+=stride) {
            float Fx = 0.0f;
            float Fy = 0.0f;
            float Fz = 0.0f;

            //read thread's old_body data
            const float old_x = in_x[i];
            const float old_y = in_y[i];
            const float old_z = in_z[i];
            const float old_vx = in_vx[i];
            const float old_vy = in_vy[i];
            const float old_vz = in_vz[i];

            int n_blocks = (n + blockDim.x - 1) / blockDim.x;
            for(int curr_block = 0; curr_block < n_blocks; ++curr_block){

                const int thread_offset = curr_block * blockDim.x + threadIdx.x;
                if(thread_offset >= n)
                    break;

                __syncthreads();
                curr_block_x[threadIdx.x] = in_x[thread_offset];
                curr_block_y[threadIdx.x] = in_y[thread_offset];
                curr_block_z[threadIdx.x] = in_z[thread_offset];
                __syncthreads();

                for(int j = 0; j < blockDim.x; ++j){
                
                    int body_idx 
                        = threadIdx.x + j < blockDim.x
                        ? threadIdx.x + j
                        : threadIdx.x + j - blockDim.x;

                    if(curr_block * blockDim.x + body_idx >= n)
                        continue;

                    const float dx = curr_block_x[body_idx] - old_x;
                    const float dy = curr_block_y[body_idx] - old_y;
                    const float dz = curr_block_z[body_idx] - old_z;
                    const float distSqr = dx*dx + dy*dy + dz*dz + NBODY_SOFTENING;
                    const float invDist = rsqrtf(distSqr);
                    const float invDist3 = invDist * invDist * invDist;
                     
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;    
                }
            }

            //compute new speed and new postion
            const float new_vx = old_vx + dt*Fx;
            const float new_vy = old_vy + dt*Fy;
            const float new_vz = old_vz + dt*Fz;
            out_vx[i] = new_vx;
            out_vy[i] = new_vy;
            out_vz[i] = new_vz;
            out_x[i] = old_x + new_vx*dt; 
            out_y[i] = old_y + new_vy*dt; 
            out_z[i] = old_z + new_vz*dt;
        }
    };

    __global__ 
    void simulation(
        unsigned int blockDimX, 
        float simTime, float dt, 
        float* in_x, float* in_y, float* in_z, float* in_vx, float* in_vy, float* in_vz,
        float* out_x, float* out_y, float* out_z, float* out_vx, float* out_vy, float* out_vz, 
        int n
    ){
        const unsigned int gridDimX = (n + blockDimX - 1) / blockDimX;
        for(float t = 0; t < simTime; t+=dt){
            kernel<<<gridDimX, blockDimX, blockDimX*sizeof(float)*3>>>(
                in_x, in_y, in_z, in_vx, in_vy, in_vz,
                out_x, out_y, out_z, out_vx, out_vy, out_vz,
                dt, n
            );
            cudaDeviceSynchronize();
            float* tmp;
            tmp = in_x; in_x = out_x; out_x = tmp;
            tmp = in_y; in_y = out_y; out_y = tmp;
            tmp = in_z; in_z = out_z; out_z = tmp;
            tmp = in_vx; in_vx = out_vx; out_vx = tmp;
            tmp = in_vy; in_vy = out_vy; out_vy = tmp;
            tmp = in_vz; in_vz = out_vz; out_vz = tmp;
        }
    }

    void NbodyCuda::run()
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        simulation<<<1, 1, 0>>>(
            blockSize,
            simulationTime, timeStep, 
            in.x, in.y, in.z, in.vx, in.vy, in.vz,
            out.x, out.y, out.z, out.vx, out.vy, out.vz,
            n
        );
        CudaKernelErrorCheck();
        cudaEventRecord(stop);
        CudaErrorCheck(cudaDeviceSynchronize());
        float kernelTime;
        cudaEventElapsedTime(&kernelTime, start, stop);
        kernelTotalTime+=kernelTime;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        simulatedTime = timeStep * ceil(simulationTime/timeStep);
    };

    std::vector<::Nbody::Body> NbodyCuda::getResult()
    {
        std::vector<::Nbody::Body> res(in.getBodiesVector());
        dataDownloadTime = in.getDataDownloadTime();
        return res; 
    };
}