#include "Swaptions/Probability.h"
#include "Swaptions/Swaptions.h"

#include <math.h>

FTYPE RanUnif(long *s)
{
    long ix, k1;
    FTYPE dRes;

    ix = *s;
    k1 = ix / 127773L;
    ix = 16807L * (ix - k1 * 127773L) - k1 * 2836L;
    if (ix < 0)
        ix = ix + 2147483647L;
    *s = ix;
    dRes = (ix * 4.656612875e-10);
    return dRes;
}

void icdf_baseline(const int N, FTYPE *in, FTYPE *out)
{

    register FTYPE z;
    register FTYPE r;

    const FTYPE a1 = -3.969683028665376e+01;
    const FTYPE a2 = 2.209460984245205e+02;
    const FTYPE a3 = -2.759285104469687e+02;
    const FTYPE a4 = 1.383577518672690e+02;
    const FTYPE a5 = -3.066479806614716e+01;
    const FTYPE a6 = 2.506628277459239e+00;

    const FTYPE b1 = -5.447609879822406e+01;
    const FTYPE b2 = 1.615858368580409e+02;
    const FTYPE b3 = -1.556989798598866e+02;
    const FTYPE b4 = 6.680131188771972e+01;
    const FTYPE b5 = -1.328068155288572e+01;

    const FTYPE c1 = -7.784894002430293e-03;
    const FTYPE c2 = -3.223964580411365e-01;
    const FTYPE c3 = -2.400758277161838e+00;
    const FTYPE c4 = -2.549732539343734e+00;
    const FTYPE c5 = 4.374664141464968e+00;
    const FTYPE c6 = 2.938163982698783e+00;

    const FTYPE d1 = 7.784695709041462e-03;
    const FTYPE d2 = 3.224671290700398e-01;
    const FTYPE d3 = 2.445134137142996e+00;
    const FTYPE d4 = 3.754408661907416e+00;

    const FTYPE u_low = 0.02425;
    const FTYPE u_high = 1.0 - u_low;

    for (int i = 0; i < N; i++){
        FTYPE u = in[i];
        if (u < u_low){
            z = sqrt(-2.0 * log(u));
            z = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0);
        } else if (u <= u_high){
            z = u - 0.5;
            r = z * z;
            z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * z / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        } else {
            z = sqrt(-2.0 * log(1.0 - u));
            z = -(((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0);
        }
        out[i] = z;
    }

    return;
}