#ifndef SGEMM_SGEMM_READWRITE
#define SGEMM_SGEMM_READWRITE

#include <Sgemm/Matrix.h>
#include <Sgemm/ReadWrite.h>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

namespace Sgemm
{
    Matrix ReadMatrixFile(std::string fileURL)
    {
        std::ifstream matrixFile(fileURL);
        if (!matrixFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        std::string lineString;
        getline(matrixFile, lineString);
        std::istringstream dimensionLineStream(lineString);
        unsigned int nrows;
        unsigned int ncols;
        dimensionLineStream >> nrows;
        dimensionLineStream >> ncols;
        
        float* data = (float*) malloc(sizeof(float)*ncols*nrows);
        for(unsigned int i = 0; i < nrows; ++i){

            getline(matrixFile, lineString);
            std::istringstream rowLineStream(lineString);

            for(unsigned int j = 0; j < ncols; ++j)
                rowLineStream >> data[i*ncols + j];
        }

        Matrix matrix(nrows, ncols, data);
        return matrix;
    };

    void WriteMatrixFile(std::string fileURL, Matrix& matrix)
    {
        std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        outFile << std::fixed << std::setprecision(6);
        outFile << matrix.nrows() << " " << matrix.ncols() << std::endl;

        for(unsigned int i = 0; i < matrix.nrows(); ++i){
            for(unsigned int j = 0; j < matrix.ncols(); ++j){
                outFile << "\t" << matrix.get(i, j);
            }
            outFile << std::endl;
        }
    };
}

#endif //SGEMM_SGEMM_READWRITE