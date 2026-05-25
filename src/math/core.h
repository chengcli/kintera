#pragma once

// C/C++
#include <cstdio>
#include <cstdlib>
#include <cstring>

// base
#include <configure.h>

namespace kintera {

/*!
 * \brief set matrix to identity matrix
 * \param[out] M[0..n*n-1]  output matrix in row-major sequential storage
 * \param[in] n            number of rows and columns in matrix M
 */
template <typename T>
DISPATCH_MACRO void set_identity(T* M, int n) {
  memset(M, 0, n * n * sizeof(T));
  for (int i = 0; i < n; ++i) M[i * n + i] = 1.0;
}

/*!
 * \brief vector-vector dot product: s = a.b
 * \param[in] a[0..n-1]  input vector
 * \param[in] b[0..n-1]  input vector
 * \param[in] n          number of elements in vectors a and b
 * \return dot product of a and b
 */
template <typename T>
DISPATCH_MACRO T vvdot(const T* a, const T* b, int n) {
  T s = 0.0;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

/*!
 * \brief vector 2-norm: ||a||
 * \param[in] a[0..n-1]  input vector
 * \param[in] n          number of elements in vector a
 * \return 2-norm of a
 */
template <typename T>
DISPATCH_MACRO T norm2(const T* a, int n) {
  return sqrt(vvdot(a, a, n));
}

/*!
 * \brief normalize vector: a = a / ||a||
 * \param[in,out] a[0..n-1]   input/output vector
 * \param[in] n               number of elements in vector a
 */
template <typename T>
DISPATCH_MACRO void normalize(T* a, int n) {
  T nrm = norm2(a, n);
  if (nrm > 0)
    for (int i = 0; i < n; ++i) a[i] /= nrm;
}

/*!
 * \brief matrix-vector dot product: y = A x
 * \param[out] y[0..n-1]  output vector
 * \param[in] A[0..n*m-1] row-major sequential storage of n x m matrix
 * \param[in] x[0..m-1]   input vector
 * \param[in] n           number of rows in matrix A
 * \param[in] m           number of columns in matrix A
 */
template <typename T>
DISPATCH_MACRO void mvdot(T* y, const T* A, const T* x, int n, int m) {
  // y = A x, A is n×m
  for (int i = 0; i < n; i++) {
    T sum = 0.0;
    for (int j = 0; j < m; j++) {
      sum += A[i * m + j] * x[j];
    }
    y[i] = sum;
  }
}

/*!
 * \brief matrix-vector dot product: y = A^T x
 * \param[out] y[0..n-1]  output vector
 * \param[in] A[0..m*n-1] row-major sequential storage of m x n matrix
 * \param[in] x[0..m-1]   input vector
 * \param[in] n           number of columns in matrix A
 * \param[in] m           number of rows in matrix A
 */
template <typename T>
DISPATCH_MACRO void mvdot_t(T* y, const T* A, const T* x, int n, int m) {
  // y = A^T x, A is m×n
  for (int i = 0; i < n; i++) {
    T sum = 0.0;
    for (int j = 0; j < m; j++) {
      sum += A[j * n + i] * x[j];
    }
    y[i] = sum;
  }
}

/*!
 * \brief matrix-matrix dot product: a.b
 * \param[out] r[0..n1*n3-1]  output matrix in row-major sequential storage
 * \param[in] a[0..n1*n2-1]   row-major sequential storage of n1 x n2 matrix
 * \param[in] b[0..n2*n3-1]   row-major sequential storage of n2 x n3 matrix
 * \param[in] n1              number of rows in matrix a
 * \param[in] n2              number of columns in matrix a (and rows in matrix
 * b)
 * \param[in] n3              number of columns in matrix b
 */
template <typename T>
DISPATCH_MACRO void mmdot(T* r, T const* a, T const* b, int n1, int n2,
                          int n3) {
  // Perform matrix multiplication
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n3; ++j) {
      T sum = 0.0;
      for (int k = 0; k < n2; ++k) {
        sum += a[i * n2 + k] * b[k * n3 + j];
      }
      r[i * n3 + j] = sum;
    }
  }
}

/*!
 * \brief matrix-matrix dot product: a.b^T
 * \param[out] r[0..n1*n3-1]  output matrix in row-major sequential storage
 * \param[in] a[0..n1*n2-1]   row-major sequential storage of n1 x n2 matrix
 * \param[in] b[0..n3*n2-1]   row-major sequential storage of n3 x n2 matrix
 * \param[in] n1              number of rows in matrix a
 * \param[in] n2              number of cols in matrix a (and cols in matrix b)
 * \param[in] n3              number of rows in matrix b
 */
template <typename T>
DISPATCH_MACRO void mmdot_t(T* r, T const* a, T const* b, int n1, int n2,
                            int n3) {
  // Perform matrix multiplication
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n3; ++j) {
      T sum = 0.0;
      for (int k = 0; k < n2; ++k) {
        sum += a[i * n2 + j] * b[j * n2 + k];
      }
      r[i * n3 + j] = sum;
    }
  }
}

/*!
 * \brief matrix-matrix dot product: ATA = A^T * A
 * \param[out] ATA[0..n*n-1]  output matrix in row-major sequential storage
 * \param[in] A[0..n*n-1]     row-major sequential storage of n x n matrix
 * \param[in] n               number of rows and columns in matrix A
 */
template <typename T>
DISPATCH_MACRO void matmul_ATA(T* ATA, const T* A, int n) {
  // ATA = A^T * A (n x n)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      T s = 0.0;
      for (int k = 0; k < n; ++k) s += A[k * n + i] * A[k * n + j];
      ATA[i * n + j] = s;
    }
}

/*!
 * \brief Solve A x = b for a dense system by Gaussian elimination with partial
 *        pivoting. Mirrors the LU-with-pivoting path of cuBLAS getrf/getrs.
 * \param[in,out] A[0..n*n-1] row-major n x n matrix, destroyed in place.
 * \param[in,out] b[0..n-1]   right-hand side on input, solution on output.
 * \param[in] n               system dimension.
 * \return true on success, false if an exactly singular pivot was found (the
 *         caller should fall back to a least-squares / pseudo-inverse solve).
 */
template <typename T>
DISPATCH_MACRO bool dsolve_lu(T* A, T* b, int n) {
  for (int k = 0; k < n; ++k) {
    // partial pivot: largest |A[i][k]| for i >= k
    int piv = k;
    T maxv = fabs(A[k * n + k]);
    for (int i = k + 1; i < n; ++i) {
      T v = fabs(A[i * n + k]);
      if (v > maxv) {
        maxv = v;
        piv = i;
      }
    }
    if (maxv == 0.0) return false;  // exactly singular column

    if (piv != k) {
      for (int j = k; j < n; ++j) {
        T t = A[k * n + j];
        A[k * n + j] = A[piv * n + j];
        A[piv * n + j] = t;
      }
      T tb = b[k];
      b[k] = b[piv];
      b[piv] = tb;
    }

    T akk = A[k * n + k];
    for (int i = k + 1; i < n; ++i) {
      T f = A[i * n + k] / akk;
      for (int j = k + 1; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
      b[i] -= f * b[k];
    }
  }

  // back substitution
  for (int i = n - 1; i >= 0; --i) {
    T s = b[i];
    for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
    b[i] = s / A[i * n + i];
  }
  return true;
}

}  // namespace kintera
