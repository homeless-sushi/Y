#include "Sgemm/Matrix.h"
#include "Sgemm/Sgemm.h"
#include "Sgemm/SgemmCuda.h"

#include <sstream>
#include <stdexcept>

#include <cmath>

namespace Sgemm
{
    __global__
    void kernel(
        float alpha,
        float beta,
        float* a,
        float* b,
        float* c,
        unsigned int m,
        unsigned int n,
        unsigned int p,
        unsigned int tileSize
    )
    {
        extern __shared__ float bCached[];

        unsigned int xCAbsIdx = blockIdx.x*tileSize + threadIdx.x;
        unsigned int yCAbsIdx = blockIdx.y*tileSize + threadIdx.y;

        float threadRes = 0;
        for(unsigned int tileIdx = 0; tileIdx < n/tileSize; ++tileIdx){

            __syncthreads();
            unsigned int xBAbsIdx = tileIdx*tileSize + threadIdx.x;
            unsigned int yBAbsIdx = yCAbsIdx;
            bCached[threadIdx.x*tileSize+threadIdx.y] = b[xBAbsIdx*p+yBAbsIdx];
            __syncthreads();

            for(unsigned int i=0; i<tileSize; ++i){
                unsigned int iOffset = 
                    (threadIdx.y + i < tileSize) ?
                    threadIdx.y + i :
                    threadIdx.y + i - tileSize;

                unsigned int xAAbsIdx = xCAbsIdx;
                unsigned int yAAbsIdx = tileIdx*tileSize + iOffset;

                threadRes += 
                    alpha * 
                    a[xAAbsIdx*n + yAAbsIdx] *
                    bCached[iOffset*tileSize + threadIdx.y];
            }
        }

        unsigned int linearizedCAbsIdx = xCAbsIdx*p + yCAbsIdx;
        c[linearizedCAbsIdx] = threadRes + beta * c[linearizedCAbsIdx];
    }

    SgemmCuda::SgemmCuda(
        float alpha,
        float beta,
        Matrix& a,
        Matrix& b,
        Matrix& c,
        unsigned int tileSize
    ):
        Sgemm(alpha, beta, a, b, c),
        tileSize_(tileSize)
    {

        if(
            a_.nrows()%tileSize_ != 0 ||
            b_.ncols()%tileSize_ != 0 ||
            c_.ncols()%tileSize_ != 0
        ){
            std::ostringstream errorMsg;
            errorMsg << "Matrix dimensions are not multiple of tile size:" << "\n"
                << "\tA is " << a.nrows() << "x"<< a.ncols() << "\n"
                << "\tB is " << b.nrows() << "x"<< b.ncols() << "\n"
                << "\tC is " << c.nrows() << "x"<< c.ncols() << "\n";
            throw std::runtime_error(errorMsg.str());
        }
        
        cudaMalloc(&aDevice_, sizeof(float)*a_.nrows()*a_.ncols());
        cudaMalloc(&bDevice_, sizeof(float)*b_.nrows()*b_.ncols());
        cudaMalloc(&cDevice_, sizeof(float)*c_.nrows()*c_.ncols());
        cudaMemcpy(aDevice_, a_.data(), sizeof(float)*a_.nrows()*a_.ncols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(bDevice_, b_.data(), sizeof(float)*b_.nrows()*b_.ncols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(cDevice_, c_.data(), sizeof(float)*c_.nrows()*c_.ncols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    SgemmCuda::~SgemmCuda()
    {
        cudaFree(aDevice_);
        cudaFree(bDevice_);
        cudaFree(cDevice_);
        aDevice_ = nullptr;
        bDevice_ = nullptr;
        cDevice_ = nullptr;
    }
    
    void SgemmCuda::run(){
        dim3 grid(c_.nrows()/tileSize_, c_.ncols()/tileSize_);
        dim3 block(tileSize_,tileSize_);
        kernel<<<grid, block, sizeof(float)*tileSize_*tileSize_>>>(
            alpha_,
            beta_,
            aDevice_,
            bDevice_,
            cDevice_,
            a_.nrows(), 
            b_.nrows(),
            c_.nrows(),
            tileSize_
        );
    }

    Matrix SgemmCuda::getResult(){
        float* resData_ = (float*) malloc(sizeof(float)*c_.nrows()*c_.ncols());
        cudaMemcpy(resData_, cDevice_, sizeof(float)*c_.nrows()*c_.ncols(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        Matrix res(c_.nrows(), c_.ncols(), resData_);
        return res;
    }
}