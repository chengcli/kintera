#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

// base
#include <configure.h>

// math
#include "lubksb.h"
#include "ludcmp.h"

#define A(i, j) a[(i) * n2 + (j)]
#define ATA(i, j) ata[(i) * n2 + (j)]
#define AUG(i, j) aug[(i) * (n2 + nact) + (j)]
#define C(i, j) c[(i) * n2 + (j)]

namespace kintera {

template <typename T>
DISPATCH_MACRO T kkt_column_scale(T objective_scale, T column_norm) {
  return column_norm > 0. ? objective_scale / column_norm : 1.;
}

template <typename T>
DISPATCH_MACRO T kkt_row_scale(T const* c, T const* d, T const* column_norm,
                               T objective_scale, int n2, int row) {
  T scale = fabs(d[row]);
  for (int j = 0; j < n2; ++j) {
    T value = fabs(c[row * n2 + j] *
                   kkt_column_scale(objective_scale, column_norm[j]));
    if (value > scale) scale = value;
  }
  return scale > 0. ? scale : 1.;
}

/*!
 * \brief solve constrained least square problem: min ||A.x - b||, s.t. C.x <= d
 *
 * This subroutine solves the constrained least square problem using the active
 * set method based on the KKT conditions. The first `neq` rows of the
 * constraint matrix `C` are treated as equality constraints, while the
 * remaining rows are treated as inequality constraints. Objective columns and
 * constraint rows are scaled internally; the returned solution is unscaled.
 *
 * \param[in,out] b[0..n1-1]    right-hand-side vector and output. Input
 *                              dimension is n1, output dimension is n2,
 * requiring n1 >= n2
 * \param[in] a[0..n1*n2-1]     row-major input matrix, A
 * \param[in] c[0..n3*n2-1]     row-major constraint matrix, C
 * \param[in] d[0..n3-1]        right-hand-side constraint vector, d
 * \param[in] n1                number of rows in matrix A
 * \param[in] n2                number of columns in matrix A
 * \param[in] n3                number of rows in matrix C
 * \param[in] neq               number of equality constraints, 0 <= neq <= n3
 * \param[in,out] max_iter      in: maximum number of iterations to perform,
 *                              out: number of iterations actually performed
 * \param[in] reg               diagonal perturbation in scaled KKT units;
 *                              zero uses regularization only if LU fails
 *
 * \return 0 on success, 1 on invalid input (e.g., neq < 0 or neq > n3),
 *         2 on failure (max_iter reached without convergence), or 3 if the
 *         KKT system remains singular or has an inconsistent zero row.
 */
template <typename T, PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO int leastsq_kkt(T* b, T const* a, T const* c, T const* d, int n1,
                               int n2, int n3, int neq, int* max_iter,
                               float reg = 0.) {
  // check if n1 > 0, n2 > 0, n3 >= 0
  if (n1 <= 0 || n2 <= 0 || n3 < 0 || n1 < n2) {
    printf(
        "Error: n1 and n2 must be positive integers and n3 >= 0, n1 >= n2.\n");
    return 1;  // invalid input
  }

  // check if 0 <= neq <= n3
  if (neq < 0 || neq > n3) {
    printf("Error: neq must be non-negative.\n");
    return 1;  // invalid input
  }

  // Allocate memory for the augmented matrix and right-hand side vector
  int size = n2 + n3;
  T *aug, *ata, *column_norm, *rhs, *eval;
  int *ct_indx, *lu_indx, *skip_row;
  size_t mark = pool_mark<Backend>();
  aug = (T*)pmalloc<Backend>(size * size * sizeof(T));
  ata = (T*)pmalloc<Backend>(n2 * n2 * sizeof(T));
  column_norm = (T*)pmalloc<Backend>(n2 * sizeof(T));
  rhs = (T*)pmalloc<Backend>(size * sizeof(T));
  eval = (T*)pmalloc<Backend>(n3 * sizeof(T));
  ct_indx = (int*)pmalloc<Backend>(n3 * sizeof(int));
  lu_indx = (int*)pmalloc<Backend>(size * sizeof(int));
  skip_row = (int*)pmalloc<Backend>(size * sizeof(int));

  T objective_scale = 1.;
  for (int i = 0; i < n1; ++i) {
    T value = fabs(b[i]);
    if (value > objective_scale) objective_scale = value;
  }

  for (int j = 0; j < n2; ++j) {
    column_norm[j] = 0.;
    for (int i = 0; i < n1; ++i) {
      T value = fabs(A(i, j));
      if (value > column_norm[j]) column_norm[j] = value;
    }
  }

  // populate A^T.A
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      ATA(i, j) = 0.0;
      for (int k = 0; k < n1; ++k) {
        T left = column_norm[i] > 0. ? A(k, i) / column_norm[i] : 0.;
        T right = column_norm[j] > 0. ? A(k, j) / column_norm[j] : 0.;
        ATA(i, j) += left * right;
      }
    }
  }

  for (int i = 0; i < n3; ++i) {
    ct_indx[i] = i;
  }

  int nactive = neq;
  int iter = 0;
  int status = 0;
  bool converged = false;
  T fallback_reg = sizeof(T) == sizeof(float) ? 1.e-5 : 1.e-10;
  T pivot_tolerance = 10. * std::numeric_limits<T>::epsilon();

  while (iter < *max_iter) {
    ++iter;
    int nact = nactive;
    bool solved = false;
    for (int attempt = 0; attempt < 2; ++attempt) {
      T primal_reg = attempt == 0 ? 0. : fallback_reg;
      T dual_reg = attempt == 0 ? reg : (reg == 0. ? -fallback_reg : reg);
      for (int i = 0; i < n2; ++i) {
        rhs[i] = 0.;
        for (int k = 0; k < n1; ++k) {
          T value = column_norm[i] > 0. ? A(k, i) / column_norm[i] : 0.;
          rhs[i] += value * (b[k] / objective_scale);
        }
        for (int j = 0; j < n2; ++j)
          AUG(i, j) = ATA(i, j) + (i == j ? primal_reg : 0.);
      }
      for (int i = 0; i < nactive; ++i) {
        int row = ct_indx[i];
        T row_scale =
            kkt_row_scale(c, d, column_norm, objective_scale, n2, row);
        for (int j = 0; j < n2; ++j) {
          T value = C(row, j) *
                    kkt_column_scale(objective_scale, column_norm[j]) /
                    row_scale;
          AUG(n2 + i, j) = value;
          AUG(j, n2 + i) = value;
        }
        rhs[n2 + i] = d[row] / row_scale;
        for (int j = 0; j < nactive; ++j)
          AUG(n2 + i, n2 + j) = i == j ? dual_reg : 0.;
      }

      for (int i = 0; i < n2 + nactive; ++i) {
        bool all_zero = true;
        for (int j = 0; j < n2 + nactive; ++j) {
          if (aug[i * (n2 + nactive) + j] != 0.0) {
            all_zero = false;
            break;
          }
        }
        skip_row[i] = all_zero;
        if (all_zero && rhs[i] != 0.) {
          status = 3;
          break;
        }
      }
      if (status != 0) break;

      if (ludcmp<T, Backend>(aug, lu_indx, n2 + nactive, skip_row,
                             pivot_tolerance) != 0) {
        lubksb(rhs, aug, lu_indx, n2 + nactive, skip_row);
        solved = true;
        break;
      }
    }
    if (!solved) {
      status = 3;
      break;
    }

    // evaluate the inactive constraints
    for (int i = nactive; i < n3; ++i) {
      int k = ct_indx[i];
      T row_scale = kkt_row_scale(c, d, column_norm, objective_scale, n2, k);
      eval[k] = -d[k] / row_scale;
      for (int j = 0; j < n2; ++j) {
        eval[k] += C(k, j) * kkt_column_scale(objective_scale, column_norm[j]) /
                   row_scale * rhs[j];
      }
    }

    for (int i = 0; i < n3; ++i) skip_row[i] = 0;
    for (int i = 0; i < nactive; ++i) skip_row[ct_indx[i]] = 1;
    int previous_active = nactive;

    // remove inactive constraints (three-way swap)
    //           mu < 0
    //           |---------------->|
    //           |<----|<----------|
    //           f     :...m       :...l
    //           |     :   |       :   |
    // | * * * | * * * * | * * * * * | x
    // |-------|---------|-----------|
    // |  EQ   |   INEQ  | INACTIVE  |
    int first = neq;
    int mid = nactive;
    int last = n3;
    while (first < mid) {
      if (rhs[n2 + first] < 0.0) {  // inactive constraint
        // swap with the last active constraint
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[mid - 1];
        ct_indx[mid - 1] = ct_indx[last - 1];
        ct_indx[last - 1] = tmp;

        T val = rhs[n2 + first];
        rhs[n2 + first] = rhs[n2 + mid - 1];
        rhs[n2 + mid - 1] = val;
        --last;
        --mid;
      } else {
        ++first;
      }
    }

    // add back inactive constraints (two-way swap)
    //                     C.x <= d
    //                     |<----->|
    //                     f       : l
    //                     |       : |
    // | * * * | * * * * | * * * * * x * |
    // |-------|---------|---------------|
    // |  EQ   |   INEQ  |   INACTIVE    |
    while (first < last) {
      int k = ct_indx[first];
      if (eval[k] > 0.) {
        // add the inactive constraint back to the active set
        ++first;
      } else {
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[last - 1];
        ct_indx[last - 1] = tmp;
        --last;
      }
    }

    nactive = first;
    converged = nactive == previous_active;
    for (int i = 0; i < nactive && converged; ++i)
      converged = skip_row[ct_indx[i]] != 0;
    if (converged) break;
  }

  if (status == 0)
    for (int i = 0; i < n2; ++i)
      b[i] = rhs[i] * kkt_column_scale(objective_scale, column_norm[i]);

  pfree<Backend>(aug);
  pfree<Backend>(ata);
  pfree<Backend>(column_norm);
  pfree<Backend>(rhs);
  pfree<Backend>(eval);
  pfree<Backend>(ct_indx);
  pfree<Backend>(lu_indx);
  pfree<Backend>(skip_row);
  pool_rewind<Backend>(mark);

  if (status != 0) {
    *max_iter = iter;
    return status;
  }

  if (!converged) {
    *max_iter = iter;
    printf("Warning: leastsq_kkt maximum number of iterations reached (%d).\n",
           *max_iter);
    return 2;  // failure to converge
  }

  *max_iter = iter;
  return 0;  // success
}

}  // namespace kintera

#undef A
#undef ATA
#undef AUG
#undef C
