#pragma once

// C/C++
#include <cstdlib>

// base
#include <configure.h>

// kintera
#include <kintera/utils/alloc.h>

// math
#include "lubksb.h"
#include "ludcmp.h"

namespace kintera {

/*!
 * \brief invert a matrix, Y = A^{-1}
 *
 * Using the backsubstitution routines,
 * it is completely straightforward to find the inverse of a matrix
 * column by column.
 *
 * \param[out] y[0..n*n-1]      row-major output matrix, Y = A^{-1}
 * \param[in,out] a[0..n*n-1]   in: row-major A matrix
 *                              out: LU-decomposed A matrix
 * \param[in] n                 size of matrix
 */
template <typename T>
DISPATCH_MACRO void luminv(T *y, T *a, int n, char *work = nullptr) {
  int *indx;
  T *col;

  if (work == nullptr) {
    indx = (int *)malloc(n * sizeof(int));
    col = (T *)malloc(n * sizeof(T));
  } else {
    indx = alloc_from<int>(work, n);
    uintptr_t p = reinterpret_cast<uintptr_t>(work);
    p = align_up(p, alignof(T));
    col = reinterpret_cast<T *>(p);
  }

  ludcmp(a, indx, n, work);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) col[i] = 0.0;
    col[j] = 1.0;
    lubksb(col, a, indx, n);
    for (int i = 0; i < n; i++) y[i * n + j] = col[i];
  }

  if (work == nullptr) {
    free(col);
    free(indx);
  }
}

}  // namespace kintera
