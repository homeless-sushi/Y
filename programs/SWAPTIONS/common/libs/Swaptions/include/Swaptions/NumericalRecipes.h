#ifndef SWAPTIONS_SWAPTIONS_NR_H
#define SWAPTIONS_SWAPTIONS_NR_H

#include "Swaptions/Swaptions.h"

int choldc(FTYPE **a, int n);
void gaussj(FTYPE **a, int n, FTYPE **b, int m);

void nrerror(const char *error_text);

int *ivector(long nl, long nh);
void free_ivector(int *v, long nl, long nh);

FTYPE *dvector( long nl, long nh );
void free_dvector( FTYPE *v, long nl, long nh );

FTYPE **dmatrix( long nrl, long nrh, long ncl, long nch );
void free_dmatrix( FTYPE **m, long nrl, long nrh, long ncl, long nch );

#endif //SWAPTIONS_SWAPTIONS_NR_H