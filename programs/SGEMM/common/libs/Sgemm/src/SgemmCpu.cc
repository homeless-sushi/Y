#include <Sgemm/Matrix.h>
#include <Sgemm/Sgemm.h>
#include <Sgemm/SgemmCpu.h>

#include <sstream>
#include <stdexcept>

namespace Sgemm
{
    SgemmCpu::SgemmCpu(
        float alpha,
        float beta,
        Matrix& a,
        Matrix& b,
        Matrix& c,
        unsigned int nThreads,
        unsigned int tileSize
    ) :
        Sgemm(
            alpha,
            beta,
            a,
            b,
            c
        ),
        nThreads_(nThreads),
        tileSize_(tileSize)
    {
        if(
            a_.nrows()%tileSize_ != 0 ||
            a_.ncols()%tileSize_ != 0 ||
            c_.ncols()%tileSize_ != 0
        ){
            std::ostringstream errorMsg;
            errorMsg << "Matrix dimensions are not multiple of tile size:" << "\n"
                << "\tA is " << a.nrows() << "x"<< a.ncols() << "\n"
                << "\tB is " << b.nrows() << "x"<< b.ncols() << "\n"
                << "\tC is " << c.nrows() << "x"<< c.ncols() << "\n";
            throw std::runtime_error(errorMsg.str());
        }
    };


    void SgemmCpu::run()
    {
        unsigned int m = a_.nrows();
        unsigned int n = a_.ncols();
        unsigned int p = b_.ncols();

        for(int ii = 0; ii < m; ii += tileSize_){
            for(int jj = 0; jj < p; jj += tileSize_){
                for(int kk = 0; kk < n; kk += tileSize_){
                    #pragma omp parallel for num_threads(nThreads_)  
                    for(int i = ii; i < ii + tileSize_; ++i){
                        for(int k = kk; k < kk + tileSize_; ++k){
                            float a_ik = a_.get(i, k);
                            for(int j = jj; j < jj + tileSize_; ++j){
                                res_.get(i, j) +=  a_ik * b_.get(k, j) * alpha_;
                            }
                        }
                    }
                }
            }
        }

        #pragma omp parallel for num_threads(nThreads_)
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                res_.get(i, j) += c_.get(i,j) * beta_;
                
    };

    Matrix SgemmCpu::getResult()
    {
        return res_;
    };
}
