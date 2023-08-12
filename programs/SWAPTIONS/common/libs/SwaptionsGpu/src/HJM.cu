#include "SwaptionsGpu/HJM.h"

#include <iostream>

#include <math.h>

#include "Swaptions/NumericalRecipes.h"
#include "Swaptions/Probability.h"
#include "Swaptions/Swaptions.h"

#include "CudaError/CudaError.h"

namespace SwaptionsGpu {
    int HJM_Swaption_Blocking(
        FTYPE *pdSwaptionPrice, // Output vector that will store simulation results in the form:
                                // Swaption Price
                                // Swaption Standard Error
        // Swaption Parameters
        FTYPE dStrike,
        FTYPE dCompounding, // Compounding convention used for quoting the strike (0 => continuous,
        // 0.5 => semi-annual, 1 => annual).
        FTYPE dMaturity,        // Maturity of the swaption (time to expiration)
        FTYPE dTenor,           // Tenor of the swap
        FTYPE dPaymentInterval, // frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
        // year
        // HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
        int iN,
        int iFactors,
        FTYPE dYears,
        FTYPE *pdYield,
        FTYPE **ppdFactors,
        // Simulation Parameters
        long iRndSeed,
        long lTrials,
        int tileSize,
        unsigned int gpuBlockSize,
        int tid,
        FTYPE *gpuPdZ,
        FTYPE *gpuRandZ
    )
    {
        int iSuccess = 0;
        int i;
        int b;  // block looping variable
        long l; // looping variables

        FTYPE ddelt = (FTYPE)(dYears / iN);                     // ddelt = HJM matrix time-step width. e.g. if dYears = 5yrs and
                                                                // iN = no. of time points = 10, then ddelt = step length = 0.5yrs
        int iFreqRatio = (int)(dPaymentInterval / ddelt + 0.5); // = ratio of time gap between swap payments and HJM step-width.
                                                                // e.g. dPaymentInterval = 1 year. ddelt = 0.5year. This implies that a swap
                                                                // payment will be made after every 2 HJM time steps.

        FTYPE dStrikeCont; // Strike quoted in continuous compounding convention.
                        // As HJM rates are continuous, the K in max(R-K,0) will be dStrikeCont and not dStrike.
        if (dCompounding == 0)
        {
            dStrikeCont = dStrike; // by convention, dCompounding = 0 means that the strike entered by user has been quoted
                                // using continuous compounding convention
        }
        else
        {
            // converting quoted strike to continuously compounded strike
            dStrikeCont = (1 / dCompounding) * log(1 + dStrike * dCompounding);
        }
        // e.g., let k be strike quoted in semi-annual convention. Therefore, 1$ at the end of
        // half a year would earn = (1+k/2). For converting to continuous compounding,
        //(1+0.5*k) = exp(K*0.5)
        //  => K = (1/0.5)*ln(1+0.5*k)

        // HJM Framework vectors and matrices
        int iSwapVectorLength; // Length of the HJM rate path at the time index corresponding to swaption maturity.

        FTYPE **ppdHJMPath; // **** per Trial data **** //

        FTYPE *pdForward;
        FTYPE **ppdDrifts;
        FTYPE *pdTotalDrift;

        // *******************************
        // ppdHJMPath = dmatrix(0,iN-1,0,iN-1);
        ppdHJMPath = dmatrix(0, iN - 1, 0, iN * tileSize - 1); // **** per Trial data **** //
        pdForward = dvector(0, iN - 1);
        ppdDrifts = dmatrix(0, iFactors - 1, 0, iN - 2);
        pdTotalDrift = dvector(0, iN - 2);

        //==================================
        // **** per Trial data **** //
        FTYPE *pdDiscountingRatePath;   // vector to store rate path along which the swaption payoff will be discounted
        FTYPE *pdPayoffDiscountFactors; // vector to store discount factors for the rate path along which the swaption
        // payoff will be discounted
        FTYPE *pdSwapRatePath;        // vector to store the rate path along which the swap payments made will be discounted
        FTYPE *pdSwapDiscountFactors; // vector to store discount factors for the rate path along which the swap
        // payments made will be discounted
        FTYPE *pdSwapPayoffs; // vector to store swap payoffs

        int iSwapStartTimeIndex;
        int iSwapTimePoints;
        FTYPE dSwapVectorYears;

        FTYPE dSwaptionPayoff;
        FTYPE dDiscSwaptionPayoff;
        FTYPE dFixedLegValue;

        // Accumulators
        FTYPE dSumSimSwaptionPrice;
        FTYPE dSumSquareSimSwaptionPrice;

        // Final returned results
        FTYPE dSimSwaptionMeanPrice;
        FTYPE dSimSwaptionStdError;

        // *******************************
        pdPayoffDiscountFactors = dvector(0, iN * tileSize - 1);
        pdDiscountingRatePath = dvector(0, iN * tileSize - 1);
        // *******************************

        iSwapVectorLength = (int)(iN - dMaturity / ddelt + 0.5); // This is the length of the HJM rate path at the time index
        // corresponding to swaption maturity.
        //  *******************************
        pdSwapRatePath = dvector(0, iSwapVectorLength * tileSize - 1);
        pdSwapDiscountFactors = dvector(0, iSwapVectorLength * tileSize - 1);
        // *******************************
        pdSwapPayoffs = dvector(0, iSwapVectorLength - 1);

        iSwapStartTimeIndex = (int)(dMaturity / ddelt + 0.5); // Swap starts at swaption maturity
        iSwapTimePoints = (int)(dTenor / ddelt + 0.5);        // Total HJM time points corresponding to the swap's tenor
        dSwapVectorYears = (FTYPE)(iSwapVectorLength * ddelt);

        // now we store the swap payoffs in the swap payoff vector
        for (i = 0; i <= iSwapVectorLength - 1; ++i)
        {
            pdSwapPayoffs[i] = 0.0; // initializing to zero
        }

        for (i = iFreqRatio; i <= iSwapTimePoints; i += iFreqRatio)
        {
            if (i != iSwapTimePoints)
            {
                pdSwapPayoffs[i] = exp(dStrikeCont * dPaymentInterval) - 1; // the bond pays coupon equal to this amount
            }

            if (i == iSwapTimePoints)
            {
                pdSwapPayoffs[i] = exp(dStrikeCont * dPaymentInterval); // at terminal time point, bond pays coupon plus par amount
            }
        }

        // generating forward curve at t=0 from supplied yield curve
        iSuccess = HJM_Yield_to_Forward(pdForward, iN, pdYield);
        if (iSuccess != 1)
        {
            return iSuccess;
        }

        // computation of drifts from factor volatilities
        iSuccess = HJM_Drifts(pdTotalDrift, ppdDrifts, iN, iFactors, dYears, ppdFactors);
        if (iSuccess != 1)
        {
            return iSuccess;
        }

        dSumSimSwaptionPrice = 0.0;
        dSumSquareSimSwaptionPrice = 0.0;

        // Simulations begin:
        for (l = 0; l <= lTrials - 1; l += tileSize)
        {
            // For each trial a new HJM Path is generated
            iSuccess = HJM_SimPath_Forward_Blocking(
                ppdHJMPath,
                iN,
                iFactors,
                dYears,
                pdForward,
                pdTotalDrift,
                ppdFactors,
                &iRndSeed,
                tileSize,
                gpuBlockSize,
                gpuPdZ,
                gpuRandZ
            );

            if (iSuccess != 1)
            {
                return iSuccess;
            }

            // now we compute the discount factor vector

            for (i = 0; i <= iN - 1; ++i)
            {
                for (b = 0; b <= tileSize - 1; b++)
                {
                    pdDiscountingRatePath[tileSize * i + b] = ppdHJMPath[i][0 + b];
                }
            }

            iSuccess = Discount_Factors_Blocking(pdPayoffDiscountFactors, iN, dYears, pdDiscountingRatePath, tileSize); /* 15% of the time goes here */

            if (iSuccess != 1)
            {
                return iSuccess;
            }

            // now we compute discount factors along the swap path
            for (i = 0; i <= iSwapVectorLength - 1; ++i)
            {
                for (b = 0; b < tileSize; b++)
                {
                    pdSwapRatePath[i * tileSize + b] =
                        ppdHJMPath[iSwapStartTimeIndex][i * tileSize + b];
                }
            }

            iSuccess = Discount_Factors_Blocking(pdSwapDiscountFactors, iSwapVectorLength, dSwapVectorYears, pdSwapRatePath, tileSize);
            if (iSuccess != 1)
            {
                return iSuccess;
            }

            // ========================
            // Simulation
            for (b = 0; b < tileSize; b++)
            {
                dFixedLegValue = 0.0;
                for (i = 0; i <= iSwapVectorLength - 1; ++i)
                {
                    dFixedLegValue += pdSwapPayoffs[i] * pdSwapDiscountFactors[i * tileSize + b];
                }
                dSwaptionPayoff = (dFixedLegValue - 1.0 < 0.) ? 0 : dFixedLegValue - 1.0;

                dDiscSwaptionPayoff = dSwaptionPayoff * pdPayoffDiscountFactors[iSwapStartTimeIndex * tileSize + b];

                // ========= end simulation ======================================

                // accumulate into the aggregating variables =====================
                dSumSimSwaptionPrice += dDiscSwaptionPayoff;
                dSumSquareSimSwaptionPrice += dDiscSwaptionPayoff * dDiscSwaptionPayoff;
            } // END BLOCK simulation
        }

        // Simulation Results Stored
        dSimSwaptionMeanPrice = dSumSimSwaptionPrice / lTrials;
        dSimSwaptionStdError =
            sqrt((dSumSquareSimSwaptionPrice - dSumSimSwaptionPrice * dSumSimSwaptionPrice / lTrials) / (lTrials - 1.0)) / sqrt((FTYPE)lTrials);

        // results returned
        pdSwaptionPrice[0] = dSimSwaptionMeanPrice;
        pdSwaptionPrice[1] = dSimSwaptionStdError;

        iSuccess = 1;
        return iSuccess;
    }

    // This function computes forward rates from supplied yield rates
    int HJM_Yield_to_Forward(
        FTYPE *pdForward, // Forward curve to be outputted
        int iN,           // Number of time-steps
        FTYPE *pdYield    // Input yield curve
    )
    {
        // forward curve computation
        pdForward[0] = pdYield[0];
        for (int i = 1; i <= iN - 1; ++i)
        {
            pdForward[i] = (i + 1) * pdYield[i] - i * pdYield[i - 1]; // as per formula
        }
        return 1;
    }

    // This function computes drift corrections required for each factor for each maturity based on given factor volatilities
    int HJM_Drifts(
        FTYPE *pdTotalDrift, // Output vector that stores the total drift correction for each maturity
        FTYPE **ppdDrifts,   // Output matrix that stores drift correction for each factor for each maturity
        int iN,
        int iFactors,
        FTYPE dYears,
        FTYPE **ppdFactors // Input factor volatilities
    )
    {
        int i, j, l; // looping variables
        FTYPE ddelt = (FTYPE)(dYears / iN);
        FTYPE dSumVol;

        // computation of factor drifts for shortest maturity
        for (i = 0; i <= iFactors - 1; ++i)
            ppdDrifts[i][0] = 0.5 * ddelt * (ppdFactors[i][0]) * (ppdFactors[i][0]);

        // computation of factor drifts for other maturities
        for (i = 0; i <= iFactors - 1; ++i)
            for (j = 1; j <= iN - 2; ++j)
            {
                ppdDrifts[i][j] = 0;
                for (l = 0; l <= j - 1; ++l)
                    ppdDrifts[i][j] -= ppdDrifts[i][l];
                dSumVol = 0;
                for (l = 0; l <= j; ++l)
                    dSumVol += ppdFactors[i][l];
                ppdDrifts[i][j] += 0.5 * ddelt * (dSumVol) * (dSumVol);
            }

        // computation of total drifts for all maturities
        for (i = 0; i <= iN - 2; ++i)
        {
            pdTotalDrift[i] = 0;
            for (j = 0; j <= iFactors - 1; ++j)
                pdTotalDrift[i] += ppdDrifts[j][i];
        }

        return 1;
    }

    // This function computes and stores an HJM Path for given inputs
    int HJM_SimPath_Forward_Blocking(
        FTYPE **ppdHJMPath,  // Matrix that stores generated HJM path (Output)
        int iN,              // Number of time-steps
        int iFactors,        // Number of factors in the HJM framework
        FTYPE dYears,        // Number of years
        FTYPE *pdForward,    // t=0 Forward curve
        FTYPE *pdTotalDrift, // Vector containing total drift corrections for different maturities
        FTYPE **ppdFactors,  // Factor volatilities
        long *lRndSeed,      // Random number seed
        int tileSize,
        unsigned int gpuBlockSize,
        FTYPE *gpuPdZ,
        FTYPE *gpuRandZ
    )
    {
        cudaError_t cudaErr; // checks return vals of CUDA functs
        int iSuccess = 0;
        int i, j, l;             // looping variables
        FTYPE **pdZ;             // vector to store random normals
        FTYPE **randZ;           // vector to store random normals
        FTYPE dTotalShock;       // total shock by which the forward curve is hit at (t, T-t)
        FTYPE ddelt, sqrt_ddelt; // length of time steps

        ddelt = (FTYPE)(dYears / iN);
        sqrt_ddelt = sqrt(ddelt);

        pdZ = dmatrix(0, iFactors - 1, 0, iN * tileSize - 1);   // assigning memory
        randZ = dmatrix(0, iFactors - 1, 0, iN * tileSize - 1); // assigning memory

        // =====================================================
        // t=0 forward curve stored iN first row of ppdHJMPath
        // At time step 0: insert expected drift
        // rest reset to 0
        for (int b = 0; b < tileSize; b++)
        {
            for (j = 0; j <= iN - 1; j++)
            {
                ppdHJMPath[0][tileSize * j + b] = pdForward[j];

                for (i = 1; i <= iN - 1; ++i)
                {
                    ppdHJMPath[i][tileSize * j + b] = 0;
                } // initializing HJMPath to zero
            }
        }
        // -----------------------------------------------------

        // =====================================================
        // sequentially generating random numbers
        for (int b = 0; b < tileSize; b++)
        {
            for (int s = 0; s < 1; s++)
            {
                for (j = 1; j <= iN - 1; ++j)
                {
                    for (l = 0; l <= iFactors - 1; ++l)
                    {
                        // compute random number in exact same sequence
                        randZ[l][tileSize * j + b + s] = RanUnif(lRndSeed); // 10% of the total executition time
                    }
                }
            }
        }

        size_t arraySize = (iN * tileSize * iFactors * sizeof(FTYPE));
        // copy data to GPU arrays
        cudaErr = cudaMemcpy(gpuPdZ, pdZ[0], arraySize, cudaMemcpyHostToDevice);
        CudaErrorCheck(cudaErr);

        cudaErr = cudaMemcpy(gpuRandZ, randZ[0], arraySize, cudaMemcpyHostToDevice);
        CudaErrorCheck(cudaErr);

        int totalThreads = (iFactors * tileSize * iN);
        int numBlocks = (int)ceil((double)totalThreads / gpuBlockSize);
        int threadsPerBlock = ((totalThreads < gpuBlockSize) ? totalThreads : gpuBlockSize);

        // =====================================================
        // shocks to hit various factors for forward curve at t
        serialB<<<numBlocks, threadsPerBlock>>>(
            gpuPdZ,
            gpuRandZ,
            totalThreads,
            lRndSeed
        );

        // check if the kernel returned an error
        cudaErr = cudaGetLastError();
        CudaErrorCheck(cudaErr);

        // Blocks until the device has completed all preceding requested tasks (make
        // sure that the device returned before continuing).
        cudaErr = cudaDeviceSynchronize();
        CudaErrorCheck(cudaErr);

        // copy back the data from the kernel
        cudaErr = cudaMemcpy(pdZ[0], gpuPdZ, arraySize, cudaMemcpyDeviceToHost);
        CudaErrorCheck(cudaErr);

        // =====================================================
        // Generation of HJM Path1
        for (int b = 0; b < tileSize; b++) // b is the blocks
        {
            for (j = 1; j <= iN - 1; ++j) // j is the timestep
            {
                for (l = 0; l <= iN - (j + 1); ++l) // l is the future steps
                {
                    dTotalShock = 0;

                    for (i = 0; i <= iFactors - 1; ++i) // i steps through the stochastic factors
                    {
                        dTotalShock += ppdFactors[i][l] * pdZ[i][tileSize * j + b];
                    }

                    ppdHJMPath[j][tileSize * l + b] = ppdHJMPath[j - 1][tileSize * (l + 1) + b] + pdTotalDrift[l] * ddelt + sqrt_ddelt * dTotalShock;
                    // as per formula
                }
            }
        } // end Blocks
        // -----------------------------------------------------

        free_dmatrix(pdZ, 0, iFactors - 1, 0, iN * tileSize - 1);
        free_dmatrix(randZ, 0, iFactors - 1, 0, iN * tileSize - 1);
        iSuccess = 1;
        return iSuccess;
    }

    __global__ 
    void serialB(
        FTYPE *pdZ,
        FTYPE *randZ,
        int totalThreads,
        long *lRndSeed
    )
    {

        int threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (threadIndex >= totalThreads)
            return;

        pdZ[threadIndex] = CumNormalInv(randZ[threadIndex]);
    }


    __device__
    FTYPE
    CumNormalInv(FTYPE u)
    {
        constexpr FTYPE a0 = 2.50662823884;
        constexpr FTYPE a1 = -18.61500062529;
        constexpr FTYPE a2 = 41.39119773534;
        constexpr FTYPE a3 = -25.44106049637;
        constexpr FTYPE b0 = -8.47351093090;
        constexpr FTYPE b1 = 23.08336743743;
        constexpr FTYPE b2 = -21.06224101826;
        constexpr FTYPE b3 = 3.13082909833;
        constexpr FTYPE c0 = 0.3374754822726147;
        constexpr FTYPE c1 = 0.9761690190917186;
        constexpr FTYPE c2 = 0.1607979714918209;
        constexpr FTYPE c3 = 0.0276438810333863;
        constexpr FTYPE c4 = 0.0038405729373609;
        constexpr FTYPE c5 = 0.0003951896511919;
        constexpr FTYPE c6 = 0.0000321767881768;
        constexpr FTYPE c7 = 0.0000002888167364;
        constexpr FTYPE c8 = 0.0000003960315187;

        // local variables
        FTYPE x, r;

        x = u - 0.5;
        if (fabs(x) < 0.42)
        {
            r = x * x;
            r = x * (((a3 * r + a2) * r + a1) * r + a0) /
                ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0);
            return r;
        } // if ( fabs(x) < 0.42 )

        r = u;
        if (x > 0.0)
        {
            r = 1.0 - u;
        } // if ( x > 0.0 )

        // ** MDS: Note that you'll need to switch to logf if you're not using floats
        r = __logf(-__logf(r));
        r = c0 + r * (c1 + r * (c2 + r * (c3 + r *
            (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))));

        if (x < 0.0)
        {
            r = -r;
        }

        return r;
    }

    int Discount_Factors_Blocking(
        FTYPE * pdDiscountFactors,
        int iN,
        FTYPE dYears,
        FTYPE *pdRatePath,
        int tileSize
    )
    {
        int i, j, b; // looping variables
        FTYPE ddelt; // HJM time-step length
        ddelt = (FTYPE)(dYears / iN);

        FTYPE *pdexpRes;
        pdexpRes = dvector(0, (iN - 1) * tileSize - 1);
        // precompute the exponientials
        for (j = 0; j <= (iN - 1) * tileSize - 1; ++j)
        {
            pdexpRes[j] = -pdRatePath[j] * ddelt;
        }
        for (j = 0; j <= (iN - 1) * tileSize - 1; ++j)
        {
            pdexpRes[j] = exp(pdexpRes[j]);
        }

        // initializing the discount factor vector
        for (i = 0; i < (iN)*tileSize; ++i)
            pdDiscountFactors[i] = 1.0;

        for (i = 1; i <= iN - 1; ++i)
        {
            for (b = 0; b < tileSize; b++)
            {
                for (j = 0; j <= i - 1; ++j)
                {
                    pdDiscountFactors[i * tileSize + b] *= pdexpRes[j * tileSize + b];
                }
            }
        }

        free_dvector(pdexpRes, 0, (iN - 1) * tileSize - 1);
        return 1;
    }
}
