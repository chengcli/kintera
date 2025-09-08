#pragma once

// C/C++
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// base
#include <configure.h>

// math
#include "swap.h"

namespace kintera {

/* Return max |A_ij| to set a scale-dependent tolerance */
template <typename T>
DISPATCH_MACRO T max_abs(const T *A, int m, int n) {
  T mx = 0.0;
  for (int i = 0; i < m * n; ++i) {
    T v = fabs(A[i]);
    if (v > mx) mx = v;
  }
  return mx;
}

/* Compute rank via row-echelon form with partial pivoting */
template <typename T>
DISPATCH_MACRO int matrix_rank(T *A, int m, int n) {
  int rank = 0;
  int row = 0;
  T scale = max_abs(A, m, n);
  /* Relative tolerance: tweak 1e-12 if needed (looser -> larger rank) */
  T eps = (scale > 0.0 ? scale : 1.0) * 1e-12;

  for (int col = 0; col < n && row < m; ++col) {
    /* Find pivot in [row..m-1] for this column */
    int piv = row;
    T piv_abs = fabs(A[piv * n + col]);
    for (int i = row + 1; i < m; ++i) {
      T v = fabs(A[i * n + col]);
      if (v > piv_abs) {
        piv_abs = v;
        piv = i;
      }
    }
    /* If no adequate pivot in this column, skip column */
    if (piv_abs <= eps) continue;

    swap_rows(A, m, n, row, piv);

    /* Eliminate entries below the pivot */
    T pivot = A[row * n + col];
    for (int i = row + 1; i < m; ++i) {
      T f = A[i * n + col] / pivot;
      if (fabs(f) <= eps) continue;
      A[i * n + col] = 0.0; /* exact zero */
      for (int j = col + 1; j < n; ++j) A[i * n + j] -= f * A[row * n + j];
    }
    ++rank;
    ++row;
  }
  return rank;
}

}  // namespace kintera
