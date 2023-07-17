#ifndef SGEMM_SGEMM_READWRITE
#define SGEMM_SGEMM_READWRITE

#include <Sgemm/Matrix.h>

#include <string>
#include <vector>

namespace Sgemm
{
    Matrix ReadMatrixFile(std::string fileURL);
    void WriteMatrixFile(std::string fileURL, Matrix& matrix);
}

#endif //SGEMM_SGEMM_READWRITE