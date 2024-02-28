#include <Sgemm/Matrix.h>

#include <algorithm>
#include <vector>

#include <cstring>

namespace Sgemm
{
    Matrix::Matrix(unsigned int nrows, unsigned int ncols, float*& data) :
        nrows_(nrows),
        ncols_(ncols),
        data_(data)
    {
        data=nullptr;
    };

    Matrix::Matrix(unsigned int nrows, unsigned int ncols) :
        nrows_(nrows),
        ncols_(ncols)
    {
        data_ = (float*) malloc(nrows_*ncols_*sizeof(float));
        memset(data_, 0, nrows_*ncols_*sizeof(float));
    };

    Matrix::Matrix(
        unsigned int nrows,
        unsigned int ncols,
        const std::vector<float>& data
    ) :
        nrows_(nrows),
        ncols_(ncols)
    {
        data_ = (float*) malloc(sizeof(float)*nrows_*ncols_);
        std::copy(data.begin(), data.end(), data_);
    };

    Matrix::Matrix(const Matrix& other) :
        nrows_(other.nrows_),
        ncols_(other.ncols_),
        data_((float*) malloc(sizeof(float)*nrows_*ncols_))
    {
        std::copy(other.data_, other.data_+(nrows_*ncols_), data_);
    };

    Matrix& Matrix::operator=(const Matrix& other)
    {
        nrows_ = other.nrows_;
        ncols_ = other.ncols_;

        data_= (float*) malloc(sizeof(float)*nrows_*ncols_);
        std::copy(other.data_, other.data_+(nrows_*ncols_), data_);

        return *this;
    };

    Matrix::Matrix(Matrix&& other) :
        nrows_(other.nrows_),
        ncols_(other.ncols_),
        data_(other.data_)
    {
        other.nrows_ = 0;
        other.ncols_ = 0;
        other.data_ = nullptr;
    };

    Matrix& Matrix::operator=(Matrix&& other)
    {
        nrows_ = other.nrows_;
        ncols_ = other.ncols_;
        data_ = other.data_;

        other.nrows_ = 0;
        other.ncols_ = 0;
        other.data_ = nullptr;

        return *this;
    };

    Matrix::~Matrix()
    {
        free(data_);
    };

    void Matrix::transpose(){ 
        float* transposed_data = (float*) malloc(sizeof(float)*ncols_*nrows_);

        for(unsigned int i = 0; i < nrows_; ++i)
            for(unsigned int j = 0; j < ncols_; ++j)
                transposed_data[j*nrows_ + i] = data_[i*ncols_ + j];

        std::swap(ncols_, nrows_);
        
        free(data_);
        data_ = transposed_data;
    };
}
