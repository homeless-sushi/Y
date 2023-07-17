#ifndef SGEMM_SGEMM_SGEMM_CPU
#define SGEMM_SGEMM_SGEMM_CPU

#include <Sgemm/Matrix.h>
#include <Sgemm/Sgemm.h>

namespace Sgemm
{
    class SgemmCpu : public ::Sgemm::Sgemm
    {
        public:
            SgemmCpu(
                float alpha,
                float beta,
                Matrix& a,
                Matrix& b,
                Matrix& c,
                unsigned int nThreads,
                unsigned int blockSize
            );

            void run();
            Matrix getResult();
        
        private:
            unsigned int nThreads_;
            unsigned int tileSize_;
    };   
}

#endif //SGEMM_SGEMM_SGEMM_CPU