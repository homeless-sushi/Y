#ifndef COMMON_CUDAERROR_CUDAERROR_H
#define COMMON_CUDAERROR_CUDAERROR_H

#include <iostream>

#include <cuda_runtime_api.h>

#define CudaErrorCheck(ans) { CudaAssert((ans), __FILE__, __LINE__); }
#define CudaKernelErrorCheck(ans) { CudaAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }

inline void CudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "CudaAssert: \n" 
         << "\t" << cudaGetErrorString(code) << "\n"
         << file << " at line " << line 
         << std::endl;
      if (abort) exit(code);
   }
}

#endif //COMMON_CUDAERROR_CUDAERROR_H