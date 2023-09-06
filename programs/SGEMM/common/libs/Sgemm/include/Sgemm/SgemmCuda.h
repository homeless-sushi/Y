#ifndef SGEMM_SGEMM_SGEMM_CUDA
#define SGEMM_SGEMM_SGEMM_CUDA

#include <Sgemm/Matrix.h>
#include <Sgemm/Sgemm.h>

namespace Sgemm
{
    class SgemmCuda : public ::Sgemm::Sgemm
    {
        public:
            SgemmCuda(
                float alpha,
                float beta,
                Matrix& a,
                Matrix& b,
                Matrix& c,
                unsigned int tileSize
            );
            ~SgemmCuda();

            void run();
            Matrix getResult();

            float getDataUploadTime() { return dataUploadTime; }
            float getKernelTime() { return kernelTime; }
            float getDataDownloadTime() { return dataDownloadTime; } 
                   
        private:
            float* aDevice_;
            float* bDevice_;
            float* cDevice_;
            unsigned int tileSize_;

            float dataUploadTime = 0;
            float kernelTime = 0;
            float dataDownloadTime = 0;
    };   
}

#endif //SGEMM_SGEMM_SGEMM_CUDA