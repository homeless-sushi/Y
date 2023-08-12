#include "SwaptionsGpu/HJM.h"
#include "SwaptionsGpu/Kernel.h"

#include <assert.h>

#include "cuda_runtime_api.h"

#include "Swaptions/Swaptions.h"

#include "CudaError/CudaError.h"

namespace SwaptionsGpu
{
    void kernel(
        parm *swaptions,
        unsigned int nSwaptions,
        unsigned int gpuBlockSize,
        unsigned int tileSize,
        unsigned int nTrials,
        long seed
    )
    {
        cudaError_t cudaErr;
        size_t arraySize = (swaptions[0].iN * tileSize * swaptions[0].iFactors * sizeof(FTYPE));

        FTYPE *gpuPdZ;
        cudaErr = cudaMalloc((void **)&gpuPdZ, arraySize);
        CudaErrorCheck(cudaErr);

        FTYPE *gpuRandZ;
        cudaErr = cudaMalloc((void **)&gpuRandZ, arraySize);
        CudaErrorCheck(cudaErr);

        for (int i = 0; i < nSwaptions; i++){
            FTYPE pdSwaptionPrice[2];
            int iSuccess = HJM_Swaption_Blocking(
                pdSwaptionPrice,
                swaptions[i].dStrike,
                swaptions[i].dCompounding,
                swaptions[i].dMaturity,
                swaptions[i].dTenor,
                swaptions[i].dPaymentInterval,
                swaptions[i].iN,
                swaptions[i].iFactors,
                swaptions[i].dYears,
                swaptions[i].pdYield,
                swaptions[i].ppdFactors,
                seed+i,
                nTrials,
                tileSize,
                gpuBlockSize,
                0,
                gpuPdZ,
                gpuRandZ
            );
            assert(iSuccess == 1);
            swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
            swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
        }

        cudaErr = cudaFree(gpuPdZ);
        CudaErrorCheck(cudaErr);
        cudaErr = cudaFree(gpuRandZ);
        CudaErrorCheck(cudaErr);
    }
}
