#include "SwaptionsCpu/HJM.h"
#include "SwaptionsCpu/Kernel.h"

#include <assert.h>

#include <omp.h>

#include "Swaptions/Swaptions.h"

namespace SwaptionsCpu
{
    void kernel(
        parm *swaptions,
        unsigned int nSwaptions,
        unsigned int nThreads,
        unsigned int tileSize,
        unsigned int nTrials,
        long seed
    )
    {
        #pragma omp parallel num_threads(nThreads) 
        {
            int threadId = omp_get_thread_num();

            int start; 
            int stop; 
            int chunkSize;

            if (threadId < (nSwaptions % nThreads)) {
                chunkSize = nSwaptions / nThreads + 1;
                start = threadId * chunkSize;
                stop = (threadId + 1) * chunkSize;
            } else {
                chunkSize = nSwaptions / nThreads;
                int offsetThread = nSwaptions % nThreads;
                int offset = offsetThread * (chunkSize + 1);
                start = offset + (threadId - offsetThread) * chunkSize;
                stop = offset + (threadId - offsetThread + 1) * chunkSize;
            }
            if (threadId == nThreads - 1)
                stop = nSwaptions;

            for (int i = start; i < stop; i++){
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
                    seed + i, 
                    nTrials, 
                    tileSize, 
                    0
                );
                assert(iSuccess == 1);
                swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
                swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
            }
        }
    }
}
