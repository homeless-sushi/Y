#ifndef SWAPTIONS_GPU_HJM_H
#define SWAPTIONS_GPU_HJM_H

#include <cuda_runtime_api.h>

#include "Swaptions/Swaptions.h"

namespace SwaptionsGpu 
{
    int HJM_Swaption_Blocking(
        FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
                                //Swaption Price
                                //Swaption Standard Error
        //Swaptions Parameters
        FTYPE dStrike,	  
        FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous, 0.5 => semi-annual, 1 => annual)
        FTYPE dMaturity,	     //Maturity of the swaption (time to expiration)
        FTYPE dTenor,	        //Tenor of the swap
        FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half year
        //HJM Framework Parameters 
        int iN,
        int iFactors,
        FTYPE dYears,
        FTYPE *pdYield,
        FTYPE **ppdFactors,
        //Simulation Parameters
        long iRndSeed,
        long lTrials,
        int tileSize,
        unsigned int gpuBlockSize,
        int tid,
        FTYPE *gpuPdZ,
        FTYPE *gpuRandZ
    );

    int HJM_Yield_to_Forward(
        FTYPE *pdForward,	//Forward curve to be outputted
        int iN,				//Number of time-steps
        FTYPE *pdYield      //Input yield curve
    );

    int HJM_Drifts(
        FTYPE *pdTotalDrift, //Output vector that stores the total drift correction for each maturity
        FTYPE **ppdDrifts,   //Output matrix that stores drift correction for each factor for each maturity
        int iN, 
        int iFactors,
        FTYPE dYears,
        FTYPE **ppdFactors //Input factor volatilities
    );

    int HJM_SimPath_Forward_Blocking(
        FTYPE **ppdHJMPath,	//Matrix that stores generated HJM path (Output)
        int iN,				//Number of time-steps
        int iFactors,		//Number of factors in the HJM framework
        FTYPE dYears,		//Number of years
        FTYPE *pdForward,	//t=0 Forward curve
        FTYPE *pdTotalDrift,//Vector containing total drift corrections for different maturities
        FTYPE **ppdFactors,	//Factor volatilities
        long *lRndSeed,		//Random number seed
        int tileSize,
        unsigned int gpuBlockSize,
        FTYPE * gpuPdZ,
        FTYPE * gpuRandZ
    );

    __global__ 
    void serialB(
        FTYPE* pdZ,
        FTYPE* randZ,
        int totalThreads,
        long* lRndSeed
    );

    __device__ 
    FTYPE CumNormalInv(FTYPE u);

    int Discount_Factors_Blocking(
        FTYPE *pdDiscountFactors, 
        int iN,
        FTYPE dYears,
        FTYPE *pdRatePath,
        int tileSize
    );
}
#endif //SWAPTIONS_GPU_HJM_H