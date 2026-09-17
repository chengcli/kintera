#pragma once

// C/C++
#include <cmath>
#include <cstring>

// base
#include <configure.h>

// kintera
#include <kintera/math/core.h>
#include <kintera/math/psolve.h>
#include <kintera/utils/alloc.h>

namespace kintera {

/*!
 * \brief Per-thread scratch size (bytes) for evolve_implicit_cell.
 *
 * Layout: A[n*n] (kept for the singular fallback) + Alu[n*n] (destroyed by the
 * LU solve) + sr[n] (rhs, kept) + x[n] (working rhs -> solution) + the psolve
 * pseudo-inverse workspace.
 */
template <typename T>
size_t evolve_implicit_space(int nspecies, int nreaction) {
  size_t bytes = 0;
  auto bump = [&](size_t, size_t nbytes) {
    bytes += pool_allocation_bytes(nbytes);
  };
  bump(alignof(T), nspecies * nspecies * sizeof(T));  // A
  bump(alignof(T), nspecies * nspecies * sizeof(T));  // Alu
  bump(alignof(T), nspecies * sizeof(T));             // sr
  bump(alignof(T), nspecies * sizeof(T));             // x
  return bytes + psolve_space<T>(nspecies);           // psolve workspace
}

/*!
 * \brief Single-cell implicit Euler kinetics solve.
 *
 * Builds A = I/dt - S*J and SR = S*rate for one cell, then solves A*delta = SR.
 * This fuses the batched stoich.matmul(jacobian) GEMM and the batched LU that
 * torch performs in evolve_implicit into one per-cell computation, eliminating
 * the cuBLAS tiny-matrix batched-GEMM and the batched getrf/getrs.
 *
 * \param[out] delta[0..nspecies-1]   concentration change for this cell.
 * \param[in]  rate[0..nreaction-1]   reaction rates for this cell.
 * \param[in]  jac[0..nreaction*nspecies-1]  reaction-space Jacobian, row-major
 *             (nreaction, nspecies).
 * \param[in]  stoich[0..nspecies*nreaction-1]  stoichiometric matrix, row-major
 *             (nspecies, nreaction). Shared (constant) across all cells.
 * \param[in]  nspecies, nreaction    system dimensions.
 * \param[in]  inv_dt                 1 / dt.
 */
template <typename T>
DISPATCH_MACRO void evolve_implicit_cell(T* delta, const T* rate, const T* jac,
                                         const T* stoich, int nspecies,
                                         int nreaction, T inv_dt,
                                         char* work = nullptr) {
  size_t mark = pool_mark(work);
  T* A = (T*)pmalloc(work, nspecies * nspecies * sizeof(T));
  T* Alu = (T*)pmalloc(work, nspecies * nspecies * sizeof(T));
  T* sr = (T*)pmalloc(work, nspecies * sizeof(T));
  T* x = (T*)pmalloc(work, nspecies * sizeof(T));

  // A = S * J  (then turned into I/dt - S*J below);  sr = S * rate
  mmdot(A, stoich, jac, nspecies, nreaction, nspecies);
  mvdot(sr, stoich, rate, nspecies, nreaction);

  for (int i = 0; i < nspecies; ++i)
    for (int k = 0; k < nspecies; ++k)
      A[i * nspecies + k] = (i == k ? inv_dt : T(0)) - A[i * nspecies + k];

  // primary path: dense LU with partial pivoting on a destroyable copy
  for (int t = 0; t < nspecies * nspecies; ++t) Alu[t] = A[t];
  for (int t = 0; t < nspecies; ++t) x[t] = sr[t];

  if (!dsolve_lu(Alu, x, nspecies)) {
    // singular cell: minimum-norm least-squares via pseudo-inverse, matching
    // the linalg_lstsq fallback in the original torch implementation.
    for (int t = 0; t < nspecies; ++t) x[t] = sr[t];
    psolve(x, A, nspecies, work);
  }

  for (int t = 0; t < nspecies; ++t) delta[t] = x[t];

  pfree(A);
  pfree(Alu);
  pfree(sr);
  pfree(x);
  pool_rewind(work, mark);
}

}  // namespace kintera
