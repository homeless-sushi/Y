#ifndef SGEMM_SGEMM_MATRIX
#define SGEMM_SGEMM_MATRIX

#include <vector>

namespace Sgemm
{
    class Matrix
    {
        public:
            Matrix(unsigned int ncols, unsigned int nrows, float*& data);
            Matrix(unsigned int ncols, unsigned int nrows);
            Matrix(unsigned int ncols, unsigned int nrows, const std::vector<float>& data);

            Matrix(const Matrix& other);
            Matrix& operator=(const Matrix& other);

            Matrix(Matrix&& other);
            Matrix& operator=(Matrix&& other);
            
            ~Matrix();

            float& get(unsigned int i, unsigned int j){ return data_[i*ncols_ + j]; };
            unsigned int nrows(){ return nrows_; };
            unsigned int ncols(){ return ncols_; };

            float* data(){ return data_; }

            void transpose();
        private:

            unsigned int ncols_;
            unsigned int nrows_;

            float* data_;
    };
}

#endif //SGEMM_SGEMM_MATRIX