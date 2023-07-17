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
        
        private:
            float* aDevice_;
            float* bDevice_;
            float* cDevice_;
            unsigned int tileSize_;
    };   
}

#endif //SGEMM_SGEMM_SGEMM_CUDA