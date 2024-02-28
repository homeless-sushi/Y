#ifndef SGEMM_SGEMM_SGEMM
#define SGEMM_SGEMM_SGEMM

#include <Sgemm/Matrix.h>

#include <vector>

namespace Sgemm
{
    class Sgemm
    {
        public:
            Sgemm(
                float alpha,
                float beta,
                Matrix& a,
                Matrix& b,
                Matrix& c
            );

            virtual void run() = 0;
            virtual Matrix getResult() = 0;

            virtual ~Sgemm() = default;

        protected:
            float alpha_;
            float beta_;
            Matrix a_;
            Matrix b_;
            Matrix c_;

            Matrix res_;
    };
}

#endif //SGEMM_SGEMM_SGEMM