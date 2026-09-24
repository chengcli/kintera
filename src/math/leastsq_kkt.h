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

// Active-set policies of the KKT driver below. Only the step that grows the
// active set differs; assembly, factorization and multiplier drops are shared.
enum class KktActivePolicy {
  // add every violated inequality at once (#110 contract of leastsq_kkt)
  AddAllViolated,
  // add the single most violated inequality that is linearly independent of
  // the active block; requires x = 0 to be feasible
  AddOneIndependent,
};

// Constraint row `row` in scaled KKT units, as assembled into the KKT block.
template <typename T>
DISPATCH_MACRO void kkt_scaled_row(T* out, T const* c, T const* d,
                                   T const* column_norm, T objective_scale,
                                   int n2, int row) {
  T row_scale = kkt_row_scale(c, d, column_norm, objective_scale, n2, row);
  for (int j = 0; j < n2; ++j)
    out[j] = c[row * n2 + j] *
             kkt_column_scale(objective_scale, column_norm[j]) / row_scale;
}

// Remove from v its components along the nq orthonormal rows of q
// (modified Gram-Schmidt, applied twice); returns the 2-norm of what remains.
template <typename T>
DISPATCH_MACRO T kkt_orthogonalize(T* v, T const* q, int nq, int n2) {
  for (int pass = 0; pass < 2; ++pass) {
    for (int r = 0; r < nq; ++r) {
      T dot = 0.;
      for (int j = 0; j < n2; ++j) dot += q[r * n2 + j] * v[j];
      for (int j = 0; j < n2; ++j) v[j] -= dot * q[r * n2 + j];
    }
  }
  T norm = 0.;
  for (int j = 0; j < n2; ++j) norm += v[j] * v[j];
  return sqrt(norm);
}

// Feasible-origin policy hook: solve the KKT system for a square A in the
// direct form [[A_s, A_s^-T C_s^T], [C_s, reg I]] [y; mu] = [b_s; d_s], the
// scaled normal equations premultiplied by A_s^-T. Same scaling, unknowns and
// multipliers as the normal-equation block, but its conditioning follows
// cond(A) instead of cond(A)^2. Pivot criterion, as in dsolve_lu (the direct
// Newton solve of constrained_newton_step): an LU (of A_s^T, then of the
// block) fails only on an exactly zero pivot, and the solution must be
// finite. The active rows are independent to sqrt(eps) by construction.
// Returns false if either fails; the caller then rebuilds aug and rhs on the
// normal-equation path.
template <typename T, PoolBackend Backend>
DISPATCH_MACRO bool kkt_solve_direct(T* aug, T* rhs, int* lu_indx, T* at_lu,
                                     T* at_col, int* at_indx, T const* a,
                                     T const* b, T const* c, T const* d,
                                     T const* column_norm, T objective_scale,
                                     int n2, int nact, int const* ct_indx,
                                     T reg, char* work) {
  int n = n2 + nact;
  for (int i = 0; i < n2; ++i)
    for (int j = 0; j < n2; ++j)
      at_lu[i * n2 + j] =
          column_norm[i] > 0. ? a[j * n2 + i] / column_norm[i] : 0.;
  if (ludcmp<T, Backend>(at_lu, at_indx, n2, nullptr, T(0), work) == 0)
    return false;

  for (int i = 0; i < n2; ++i) {
    rhs[i] = b[i] / objective_scale;
    for (int j = 0; j < n2; ++j)
      aug[i * n + j] =
          column_norm[j] > 0. ? a[i * n2 + j] / column_norm[j] : 0.;
  }
  for (int k = 0; k < nact; ++k) {
    int row = ct_indx[k];
    kkt_scaled_row(aug + (n2 + k) * n, c, d, column_norm, objective_scale, n2,
                   row);
    rhs[n2 + k] =
        d[row] / kkt_row_scale(c, d, column_norm, objective_scale, n2, row);
    for (int j = 0; j < n2; ++j) at_col[j] = aug[(n2 + k) * n + j];
    lubksb(at_col, at_lu, at_indx, n2);
    for (int j = 0; j < n2; ++j) aug[j * n + n2 + k] = at_col[j];
    for (int j = 0; j < nact; ++j)
      aug[(n2 + k) * n + n2 + j] = j == k ? reg : 0.;
  }

  if (ludcmp<T, Backend>(aug, lu_indx, n, nullptr, T(0), work) == 0)
    return false;
  lubksb(rhs, aug, lu_indx, n);
  for (int i = 0; i < n; ++i)
    if (!std::isfinite(rhs[i])) return false;
  return true;
}

namespace detail {

template <typename T, PoolBackend Backend, KktActivePolicy Policy>
DISPATCH_MACRO int leastsq_kkt_impl(T* b, T const* a, T const* c, T const* d,
                                    int n1, int n2, int n3, int neq,
                                    int* max_iter, float reg, char* work) {
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
  size_t mark = pool_mark<Backend>(work);
  aug = (T*)pmalloc<Backend>(work, size * size * sizeof(T));
  ata = (T*)pmalloc<Backend>(work, n2 * n2 * sizeof(T));
  column_norm = (T*)pmalloc<Backend>(work, n2 * sizeof(T));
  rhs = (T*)pmalloc<Backend>(work, size * sizeof(T));
  eval = (T*)pmalloc<Backend>(work, n3 * sizeof(T));
  ct_indx = (int*)pmalloc<Backend>(work, n3 * sizeof(int));
  lu_indx = (int*)pmalloc<Backend>(work, size * sizeof(int));
  skip_row = (int*)pmalloc<Backend>(work, size * sizeof(int));
  // direct-solve hook buffers, feasible-origin policy only
  T *at_lu = nullptr, *at_col = nullptr;
  int* at_indx = nullptr;
  if (Policy == KktActivePolicy::AddOneIndependent) {
    at_lu = (T*)pmalloc<Backend>(work, n2 * n2 * sizeof(T));
    at_col = (T*)pmalloc<Backend>(work, n2 * sizeof(T));
    at_indx = (int*)pmalloc<Backend>(work, n2 * sizeof(int));
  }

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
  bool used_regularization = false;
  T fallback_reg = sizeof(T) == sizeof(float) ? 1.e-5 : 1.e-10;
  T pivot_tolerance = 10. * std::numeric_limits<T>::epsilon();
  // A candidate row is independent of the active block if, in scaled KKT
  // units, the part of it orthogonal to the active rows keeps at least this
  // fraction of its 2-norm. The normal-equation KKT block's condition number
  // grows like 1 / sigma_min(C_active)^2, so rows closer to dependence than
  // sqrt(eps) would leave no significant digits in the multipliers.
  T independence_tolerance = sqrt(std::numeric_limits<T>::epsilon());

  if (Policy == KktActivePolicy::AddOneIndependent) {
    // The policy never activates a dependent row, which is only safe if x = 0
    // satisfies every row: d >= 0 on inequalities and d = 0 on equalities.
    // Bounds are compared with the largest |d|, not row by row: a bound that
    // is the round-off residue of a depleted stock (-1e-19 after using up
    // 1e-3) is feasible, even when that row's own scale is as small.
    T bound_scale = 0.;
    for (int row = 0; row < n3; ++row)
      if (fabs(d[row]) > bound_scale) bound_scale = fabs(d[row]);
    T origin_tolerance = 64. * std::numeric_limits<T>::epsilon() * bound_scale;
    for (int row = 0; row < n3; ++row) {
      if (!(row < neq ? fabs(d[row]) <= origin_tolerance
                      : d[row] >= -origin_tolerance)) {
        printf("Error: x = 0 violates constraint %d (d = %g, max |d| = %g).\n",
               row, (double)d[row], (double)bound_scale);
        status = 1;  // invalid input
        break;
      }
    }
  }

  while (status == 0 && iter < *max_iter) {
    ++iter;
    int nact = nactive;
    bool solved = false;
    // feasible-origin policy hook: a square A is solved in the direct form
    // first; the normal-equation attempts below are its fallback
    if (Policy == KktActivePolicy::AddOneIndependent && n1 == n2)
      solved = kkt_solve_direct<T, Backend>(
          aug, rhs, lu_indx, at_lu, at_col, at_indx, a, b, c, d, column_norm,
          objective_scale, n2, nactive, ct_indx, T(reg), work);
    for (int attempt = 0; attempt < 2 && !solved; ++attempt) {
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
                             pivot_tolerance, work) != 0) {
        lubksb(rhs, aug, lu_indx, n2 + nactive, skip_row);
        used_regularization = primal_reg != 0. || dual_reg != 0.;
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

    if (Policy == KktActivePolicy::AddAllViolated) {
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
    } else {
      // add back ONE constraint: the most violated row that is independent of
      // the active block. aug is free until the next factorization: q holds an
      // orthonormal basis of the scaled active rows, v the candidate row.
      T* q = aug;
      T* v = aug + n3 * n2;
      int nq = 0;
      for (int i = 0; i < first; ++i) {
        T* row = q + nq * n2;
        kkt_scaled_row(row, c, d, column_norm, objective_scale, n2, ct_indx[i]);
        T norm = 0.;
        for (int j = 0; j < n2; ++j) norm += row[j] * row[j];
        norm = sqrt(norm);
        T remain = kkt_orthogonalize(row, q, nq, n2);
        if (remain > independence_tolerance * norm) {
          for (int j = 0; j < n2; ++j) row[j] /= remain;
          ++nq;
        }
      }

      int best = -1;
      T worst = 0.;
      for (int i = first; i < last; ++i) {
        int k = ct_indx[i];
        if (!(eval[k] > worst)) continue;
        kkt_scaled_row(v, c, d, column_norm, objective_scale, n2, k);
        T norm = 0.;
        for (int j = 0; j < n2; ++j) norm += v[j] * v[j];
        norm = sqrt(norm);
        if (kkt_orthogonalize(v, q, nq, n2) > independence_tolerance * norm) {
          worst = eval[k];
          best = i;
        }
      }
      if (best >= 0) {
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[best];
        ct_indx[best] = tmp;
        ++first;
      }
    }

    nactive = first;
    converged = nactive == previous_active;
    for (int i = 0; i < nactive && converged; ++i)
      converged = skip_row[ct_indx[i]] != 0;
    if (converged) break;
  }

  if (status == 0 && converged && used_regularization) {
    T feasibility_tolerance =
        64. * std::numeric_limits<T>::epsilon() + 10. * fallback_reg;
    // a dependent row left violated is part of the AddOneIndependent contract
    int ncheck = Policy == KktActivePolicy::AddAllViolated ? n3 : nactive;
    for (int i = 0; i < ncheck; ++i) {
      int row = ct_indx[i];
      T row_scale = kkt_row_scale(c, d, column_norm, objective_scale, n2, row);
      T residual = -d[row] / row_scale;
      for (int j = 0; j < n2; ++j) {
        residual += C(row, j) *
                    kkt_column_scale(objective_scale, column_norm[j]) /
                    row_scale * rhs[j];
      }
      if (!std::isfinite(residual) ||
          (i < nactive ? fabs(residual) > feasibility_tolerance
                       : residual > feasibility_tolerance)) {
        status = 3;
        break;
      }
    }
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
  if (Policy == KktActivePolicy::AddOneIndependent) {
    pfree<Backend>(at_lu);
    pfree<Backend>(at_col);
    pfree<Backend>(at_indx);
  }
  pool_rewind<Backend>(work, mark);

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

}  // namespace detail

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
 *         KKT system remains singular or its constraints are infeasible.
 */
template <typename T, PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO int leastsq_kkt(T* b, T const* a, T const* c, T const* d, int n1,
                               int n2, int n3, int neq, int* max_iter,
                               float reg = 0., char* work = nullptr) {
  return detail::leastsq_kkt_impl<T, Backend, KktActivePolicy::AddAllViolated>(
      b, a, c, d, n1, n2, n3, neq, max_iter, reg, work);
}

/*!
 * \brief leastsq_kkt for problems where x = 0 is feasible (C.0 <= d)
 *
 * Same problem, arguments, scaling, KKT assembly and factorization as
 * leastsq_kkt; only the active-set policy differs. The policy also tries,
 * for a square A, the direct KKT form first (kkt_solve_direct), which avoids
 * squaring cond(A); if it hits a zero pivot or a non-finite solution the
 * shared normal-equation path runs as in leastsq_kkt. Each iteration adds back
 * the single most violated inequality that is linearly independent of the
 * active block, so the active block never goes rank-deficient when several
 * rows share a direction (e.g. two condensates drawing on one vapour).
 *
 * A row counts as independent when the part of its scaled form orthogonal to
 * the active rows keeps more than sqrt(eps) of its 2-norm (relative, so it
 * does not depend on the magnitude of C or d).
 *
 * A violated row that is dependent on the active block is never activated,
 * so the returned x may violate it; the caller must limit the step (the
 * saturation adjustment clips each reaction extent to its reactant stock).
 * This is only meaningful when x = 0 is feasible, which is checked on entry:
 * d >= -64 eps max|d| and, for the first neq rows, |d| <= 64 eps max|d|, so
 * the bounds should share units (the callers pass concentrations).
 *
 * Callers: equilibrate_uv and equilibrate_tp, whose bounds are the current
 * concentrations. A general QP must use leastsq_kkt, which fails closed.
 *
 * \return 0 on success, 1 on invalid input or if x = 0 is infeasible (b is
 *         left unchanged), 2 if max_iter is reached without convergence, or
 *         3 if the KKT system remains singular.
 */
template <typename T, PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO int leastsq_kkt_feasible_origin(T* b, T const* a, T const* c,
                                               T const* d, int n1, int n2,
                                               int n3, int neq, int* max_iter,
                                               float reg = 0.,
                                               char* work = nullptr) {
  return detail::leastsq_kkt_impl<T, Backend,
                                  KktActivePolicy::AddOneIndependent>(
      b, a, c, d, n1, n2, n3, neq, max_iter, reg, work);
}

}  // namespace kintera

#undef A
#undef ATA
#undef AUG
#undef C
