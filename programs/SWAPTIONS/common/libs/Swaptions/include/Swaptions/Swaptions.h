#ifndef SWAPTIONS_SWAPTIONS_SWAPTIONS_H
#define SWAPTIONS_SWAPTIONS_SWAPTIONS_H

#define FTYPE double
#define BLOCK_SIZE 16 // Blocking to allow better caching

typedef struct
{
  int Id;
  FTYPE dSimSwaptionMeanPrice;
  FTYPE dSimSwaptionStdError;
  FTYPE dStrike;
  FTYPE dCompounding;
  FTYPE dMaturity;
  FTYPE dTenor;
  FTYPE dPaymentInterval;
  int iN;
  FTYPE dYears;
  int iFactors;
  FTYPE *pdYield;
  FTYPE **ppdFactors;
} parm;

#endif //SWAPTIONS_SWAPTIONS_SWAPTIONS_H
