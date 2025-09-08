#pragma once

// C/C++
#include <cstdio>
#include <cstdlib>

// base
#include <configure.h>

// kintera
#include <kintera/utils/alloc.h>

// math
#include "lubksb.h"
#include "ludcmp.h"
#include "psolve.h"

/*
 * Solve kkt-block system:
 *   [ A  B^T ] [ x ] = [ C ]
 *   [ B   0  ] [ y ]   [ D ]
 *
 * A: n x n symmetric matrix (invertible)
 * B: m x n matrix
 * C: n-vector
 * D: m-vector
 *
 * Output:
 *   x (n-vector), y (m-vector)
 */

namespace kintera {

// matrix utilities (row-major)
template <typename T>
DISPATCH_MACRO void matvec(T *y, const T *A, const T *x, int n, int m) {
  // y = A x, A is n×m
  for (int i = 0; i < n; i++) {
    T sum = 0.0;
    for (int j = 0; j < m; j++) {
      sum += A[i * m + j] * x[j];
    }
    y[i] = sum;
  }
}

template <typename T>
DISPATCH_MACRO void matvec_t(T *y, const T *A, const T *x, int n, int m) {
  // y = A^T x, A is m×n
  for (int i = 0; i < n; i++) {
    T sum = 0.0;
    for (int j = 0; j < m; j++) {
      sum += A[j * n + i] * x[j];
    }
    y[i] = sum;
  }
}

template <typename T>
DISPATCH_MACRO void matmat(T *C, const T *A, const T *B, int n, int m, int p) {
  // C = A B, A is n×m, B is m×p, C is n×p
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      T sum = 0.0;
      for (int k = 0; k < m; k++) {
        sum += A[i * m + k] * B[k * p + j];
      }
      C[i * p + j] = sum;
    }
  }
}

template <typename T>
DISPATCH_MACRO void matmat_t(T *C, const T *A, const T *B, int n, int m,
                             int p) {
  // C = A B, A is n×m, B is p×m, C is n×p
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      T sum = 0.0;
      for (int k = 0; k < m; k++) {
        sum += A[i * m + k] * B[j * m + k];
      }
      C[i * p + j] = sum;
    }
  }
}

template <typename T>
DISPATCH_MACRO void solve_block_system(const T *A_inv, const T *B, T *C, T *D,
                                       int n, int m, float ftol = 1.e-10,
                                       char *work = nullptr) {
  T *B_Ainv, *B_Ainv_Bt, *tmp_n, *B_Ainv_C;
  int *indx;

  if (work == nullptr) {
    B_Ainv = (T *)malloc(m * n * sizeof(T));
    B_Ainv_Bt = (T *)malloc(m * m * sizeof(T));
    tmp_n = (T *)malloc(n * sizeof(T));
    B_Ainv_C = (T *)malloc(m * sizeof(T));
    indx = (int *)malloc(m * sizeof(int));
  } else {
    B_Ainv = alloc_from<T>(work, m * n);
    B_Ainv_Bt = alloc_from<T>(work, m * m);
    tmp_n = alloc_from<T>(work, n);
    B_Ainv_C = alloc_from<T>(work, m);
    indx = alloc_from<int>(work, m);
  }

  // Step 1: Compute B_Ainv and B_Ainv_Bt
  matmat(B_Ainv, B, A_inv, m, n, n);  // (m×n)(n×n) = m×n
  matmat_t(B_Ainv_Bt, B_Ainv, B, m, n, m);

  // Step 2: Compute y = B_Ainv*C - D
  matvec(tmp_n, A_inv, C, n, n);     // tmp_n = A^{-1} C
  matvec(B_Ainv_C, B, tmp_n, m, n);  // B_Ainv_C = B A^{-1} C

  for (int i = 0; i < m; i++) {
    D[i] = B_Ainv_C[i] - D[i];
  }

  // Step 3: Solve (B_Ainv_Bt) y = rhs_y
  /*printf("D = \n");
  for (int i = 0; i < m; i++) {
    printf("%f ", D[i]);
  }

  printf("\nB_Ainv_Bt = \n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      printf("%f ", B_Ainv_Bt[i * m + j]);
    }
    printf("\n");
  }*/

  psolve(D, B_Ainv_Bt, m, ftol, work);

  // Step 4: Recover x = A^{-1}(C - B^T y)
  matvec_t(tmp_n, B, D, n, m);
  for (int i = 0; i < n; i++) {
    tmp_n[i] = C[i] - tmp_n[i];
  }
  matvec(C, A_inv, tmp_n, n, n);

  // Free workspace
  if (work == nullptr) {
    free(B_Ainv);
    free(B_Ainv_Bt);
    free(tmp_n);
    free(B_Ainv_C);
    free(indx);
  }
}

}  // namespace kintera
