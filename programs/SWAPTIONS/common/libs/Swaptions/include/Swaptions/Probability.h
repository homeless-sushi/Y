#ifndef SWAPTIONS_SWAPTIONS_PROBABILITY_H
#define SWAPTIONS_SWAPTIONS_PROBABILITY_H

#include "Swaptions/Swaptions.h"

FTYPE RanUnif(long *s);
void icdf_baseline(const int N, FTYPE *in, FTYPE *out);

#endif //SWAPTIONS_SWAPTIONS_PROBABILITY_H
