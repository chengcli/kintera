#pragma once

// C/C++
#include <cstdlib>

// base
#include <configure.h>

// kintera
#include "lubksb.h"

namespace kintera {

/*!
 * \brief invert a matrix, Y = A^{-1}
 *
 * Using the backsubstitution routines,
 * it is completely straightforward to find the inverse of a matrix
 * column by column.
 *
 * \param[out] y[0..n*n-1] row-major output matrix, Y = A^{-1}
 * \param[in] a[0..n*n-1] row-major LU decomposition matrix
 * \param[in] indx[0..n-1] the permutation vector returned by ludcmp.
 * \param[in] n size of matrix
 */
template <typename T>
DISPATCH_MACRO void luminv(T *y, T const *a, int const *indx, int n) {
  T *col = (T *)malloc(n * sizeof(T));
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) col[i] = 0.0;
    col[j] = 1.0;
    lubksb(col, a, indx, n);
    for (int i = 0; i < n; i++) y[i * n + j] = col[i];
  }
  free(col);
}

}  // namespace kintera
