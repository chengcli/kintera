#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// base
#include <configure.h>

// math
#include "core.h"
#include "swap.h"

// kintera
#include <kintera/utils/alloc.h>

namespace kintera {

template <typename T>
DISPATCH_MACRO T max_abs_offdiag(const T* S, int n, int* p, int* q) {
  T maxv = 0.0;
  *p = 0;
  *q = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      T v = fabs(S[i * n + j]);
      if (v > maxv) {
        maxv = v;
        *p = i;
        *q = j;
      }
    }
  }
  return maxv;
}

/* Jacobi eigen-decomposition for real symmetric matrices.
   Input:  S (n x n) symmetric
   Output: eigenvalues in evals, eigenvectors in V (columns)
   S is overwritten during iterations.
*/
template <typename T>
DISPATCH_MACRO void jacobi_eigen_symmetric(T* S, int n, T* evals, T* V) {
  set_identity(V, n);
  const int max_sweeps = 100;  // sufficient for moderate n
  const T eps = 1e-12;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    int p, q;
    T off = max_abs_offdiag(S, n, &p, &q);
    if (off < eps) break;

    T app = S[p * n + p];
    T aqq = S[q * n + q];
    T apq = S[p * n + q];

    if (fabs(apq) < eps) continue;

    T tau = (aqq - app) / (2.0 * apq);
    T t = (tau >= 0.0) ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                       : 1.0 / (tau - sqrt(1.0 + tau * tau));
    T c = 1.0 / sqrt(1.0 + t * t);
    T s = t * c;

    // Update S = J^T S J
    for (int k = 0; k < n; ++k) {
      if (k != p && k != q) {
        T skp = S[k * n + p];
        T skq = S[k * n + q];
        S[k * n + p] = c * skp - s * skq;
        S[p * n + k] = S[k * n + p];
        S[k * n + q] = s * skp + c * skq;
        S[q * n + k] = S[k * n + q];
      }
    }
    T new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    T new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    S[p * n + p] = new_app;
    S[q * n + q] = new_aqq;
    S[p * n + q] = 0.0;
    S[q * n + p] = 0.0;

    // Update V = V J
    for (int k = 0; k < n; ++k) {
      T vkp = V[k * n + p];
      T vkq = V[k * n + q];
      V[k * n + p] = c * vkp - s * vkq;
      V[k * n + q] = s * vkp + c * vkq;
    }
  }

  for (int i = 0; i < n; ++i) evals[i] = S[i * n + i];
}

/* Sort eigenpairs (V columns and evals) by descending evals */
template <typename T>
DISPATCH_MACRO void sort_eigenpairs_desc(T* evals, T* V, int n) {
  for (int i = 0; i < n - 1; ++i) {
    int maxj = i;
    for (int j = i + 1; j < n; ++j)
      if (evals[j] > evals[maxj]) maxj = j;
    if (maxj != i) {
      swap_vals(evals, i, maxj);
      swap_cols_safe(V, n, i, maxj);
    }
  }
}

/* Solve x = A^+ b using SVD via eigen-decomp of A^T A.
   A: n x n, b: n, output x: n
*/
template <typename T>
DISPATCH_MACRO void psolve(T* b, const T* A, int n, char* work = nullptr) {
  T *ATA, *V, *eval, *vi, *Avi, *b0;

  if (work == nullptr) {
    ATA = (T*)malloc(n * n * sizeof(T));
    V = (T*)malloc(n * n * sizeof(T));
    eval = (T*)malloc(n * sizeof(T));
    vi = (T*)malloc(n * sizeof(T));
    Avi = (T*)malloc(n * sizeof(T));
    b0 = (T*)malloc(n * sizeof(T));
  } else {
    ATA = alloc_from<T>(work, n * n);
    V = alloc_from<T>(work, n * n);
    eval = alloc_from<T>(work, n);
    vi = alloc_from<T>(work, n);
    Avi = alloc_from<T>(work, n);
    b0 = alloc_from<T>(work, n);
  }

  matmul_ATA(ATA, A, n);
  memcpy(b0, b, n * sizeof(T));
  memset(b, 0, n * sizeof(T));  // output x

  jacobi_eigen_symmetric(ATA, n, eval, V);
  // Clean tiny negative due to roundoff and sort
  for (int i = 0; i < n; ++i) {
    if (eval[i] < 0 && eval[i] > -1e-14) eval[i] = 0.0;
  }
  sort_eigenpairs_desc(eval, V, n);

  // Tolerance relative to the largest eigenvalue (sigma^2)
  T maxlam = (n > 0) ? eval[0] : 0.0;
  T tol = (maxlam > 0 ? maxlam : 1.0) * 1e-12;

  // x = sum_i ( (u_i^T b)/sigma_i ) * v_i, where u_i = (A v_i)/sigma_i
  for (int i = 0; i < n; ++i) {
    T lam = eval[i];
    if (lam <= tol) continue;  // treat as zero singular value
    T sigma = sqrt(lam);

    // vi = column i of V
    for (int k = 0; k < n; ++k) vi[k] = V[k * n + i];

    // Avi = A * vi
    mvdot(Avi, A, vi, n, n);

    // u_i = (A v_i) / sigma
    for (int k = 0; k < n; ++k) Avi[k] /= sigma;

    // coeff = (u_i^T b) / sigma
    T coeff = vvdot(Avi, b0, n) / sigma;

    // x += coeff * v_i
    for (int k = 0; k < n; ++k) b[k] += coeff * vi[k];
  }

  if (work == nullptr) {
    free(ATA);
    free(V);
    free(eval);
    free(vi);
    free(Avi);
    free(b0);
  }
}

}  // namespace kintera
