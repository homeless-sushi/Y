#include "Swaptions/NumericalRecipes.h"
#include "Swaptions/Swaptions.h"

#include <iostream>

#include <math.h>
#include <stdlib.h>

void nrerror(const char *error_text)
{
	std::cerr << "Numerical Recipes Error:" << std::endl;
	std::cerr << "\t" << error_text << std::endl;
	exit(1);
}

// Check if a is positive definitive
int choldc(FTYPE **a, int n)
{
	FTYPE sum;

	for (int i = 1; i <= n; i++)
	{
		for (int j = i; j <= n; j++)
		{

			sum = a[i][j];

			for (int k = i - 1; k >= 1; k--)
				sum -= a[i][k] * a[j][k];

			if (i == j)
			{
				if (sum <= 0.0)
					return 0;
				a[i][i] = sqrt(sum);
			}
			else
			{
				a[j][i] = sum / a[i][i];
			}
		}
	}

	for (int i = 1; i <= n - 1; i++)
		for (int j = i + 1; j <= n; j++)
			a[i][j] = 0.0;

	return 1;
}

void gaussj(FTYPE **a, int n, FTYPE **b, int m)
{
#define SWAP(a, b) { temp = (a); (a) = (b); (b) = temp; }

	int *indxc, *indxr, *ipiv;
	int i, icol, irow, j, k, l, ll;
	FTYPE big, dum, pivinv, temp;

	indxc = ivector(1, n);
	indxr = ivector(1, n);
	ipiv = ivector(1, n);
	icol = -1;
	irow = -1;
	for (j = 1; j <= n; j++)
		ipiv[j] = 0;
	for (i = 1; i <= n; i++)
	{
		big = 0.0;
		for (j = 1; j <= n; j++)
			if (ipiv[j] != 1)
				for (k = 1; k <= n; k++)
				{
					if (ipiv[k] == 0)
					{
						if (fabs(a[j][k]) >= big)
						{
							big = fabs(a[j][k]);
							irow = j;
							icol = k;
						}
					}
					else if (ipiv[k] > 1)
						nrerror("gaussj: Singular Matrix-1");
				}
		++(ipiv[icol]);
		if (irow != icol)
		{
			for (l = 1; l <= n; l++)
				SWAP(a[irow][l], a[icol][l])
			for (l = 1; l <= m; l++)
				SWAP(b[irow][l], b[icol][l])
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if (a[icol][icol] == 0.0)
			nrerror("gaussj: Singular Matrix-2");
		pivinv = 1.0 / a[icol][icol];
		a[icol][icol] = 1.0;
		for (l = 1; l <= n; l++)
			a[icol][l] *= pivinv;
		for (l = 1; l <= m; l++)
			b[icol][l] *= pivinv;
		for (ll = 1; ll <= n; ll++)
			if (ll != icol)
			{
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				for (l = 1; l <= n; l++)
					a[ll][l] -= a[icol][l] * dum;
				for (l = 1; l <= m; l++)
					b[ll][l] -= b[icol][l] * dum;
			}
	}
	for (l = n; l >= 1; l--)
	{
		if (indxr[l] != indxc[l])
			for (k = 1; k <= n; k++)
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
	}

	free_ivector(ipiv, 1, n);
	free_ivector(indxr, 1, n);
	free_ivector(indxc, 1, n);
#undef SWAP
}

int *ivector(long nl, long nh)
{
	int *v;

	v = (int *)malloc((size_t)((nh - nl + 2) * sizeof(int)));
	if (!v)
		nrerror("allocation failure in ivector()");
	return v - nl + 1;
}

void free_ivector(int *v, long nl, long nh)
{
	free((char *)(v + nl - 1));
}

FTYPE *dvector(long nl, long nh)
{
	FTYPE *v;

	v = (FTYPE *)malloc((size_t)((nh - nl + 2) * sizeof(FTYPE)));
	if (!v)
		nrerror("allocation failure in dvector()");
	return v - nl + 1;
}

void free_dvector(FTYPE *v, long nl, long nh)
{
	free((char *)(v + nl - 1));
}

FTYPE **dmatrix(long nrl, long nrh, long ncl, long nch)
{

	long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	FTYPE **m;

	m = (FTYPE **)malloc((size_t)((nrow + 1) * sizeof(FTYPE *)));
	if (!m)
		nrerror("allocation failure 1 in dmatrix()");
	m += 1;
	m -= nrl;

	m[nrl] = (FTYPE *)malloc((size_t)((nrow * ncol + 1) * sizeof(FTYPE)));
	if (!m[nrl])
		nrerror("allocation failure 2 in dmatrix()");
	m[nrl] += 1;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++)
		m[i] = m[i - 1] + ncol;

	return m;
}

void free_dmatrix(FTYPE **m, long nrl, long nrh, long ncl, long nch)
{
	free((char *)(m[nrl] + ncl - 1));
	free((char *)(m + nrl - 1));
}
