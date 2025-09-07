#pragma once

// C/C++
#include <cstddef>
#include <cstdint>
#include <type_traits>

// base
#include <configure.h>

namespace kintera {

DISPATCH_MACRO inline uintptr_t align_up(uintptr_t p, size_t a) {
  // a must be power of two; works for 4, 8, 16, ...
  return (p + (a - 1)) & ~(a - 1);
}

template <typename U>
DISPATCH_MACRO inline U* alloc_from(char*& cursor, size_t count) {
  uintptr_t p = reinterpret_cast<uintptr_t>(cursor);
  p = align_up(p, alignof(U));
  U* out = reinterpret_cast<U*>(p);
  cursor = reinterpret_cast<char*>(p + count * sizeof(U));
  return out;
}

template <typename T>
size_t ludcmp_space(int n) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };

  bump(alignof(T), n * sizeof(T));  // vv
  return bytes;
}

template <typename T>
size_t luminv_space(int n) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };

  bump(alignof(int), n * sizeof(int));  // indx
  return bytes;
}

template <typename T>
size_t leastsq_space(int n1, int n2) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };

  bump(alignof(T), n1 * sizeof(T));       // c
  bump(alignof(T), n2 * n2 * sizeof(T));  // y
  bump(alignof(int), n2 * sizeof(int));   // indx
  return bytes + ludcmp_space<T>(n2);
}

template <typename T>
size_t block_system_space(int n, int m) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };

  bump(alignof(T), m * n * sizeof(T));  // B_Ainv
  bump(alignof(T), m * m * sizeof(T));  // B_Ainv_Bt
  bump(alignof(T), n * sizeof(T));      // tmp_n
  bump(alignof(T), m * sizeof(T));      // B_Ainv_C
  return bytes + leastsq_space<T>(n, n);
}

template <typename T>
size_t leastsq_kkt_space(int n2, int n3) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };

  bump(alignof(T), n2 * n2 * sizeof(T));    // ata
  bump(alignof(T), n2 * sizeof(T));         // atb
  bump(alignof(T), n2 * n2 * sizeof(T));    // ata_inv
  bump(alignof(T), (n2 + n3) * sizeof(T));  // rhs
  bump(alignof(T), n3 * sizeof(T));         // eval
  bump(alignof(int), n3 * sizeof(int));     // ct_indx
  return bytes + luminv_space<T>(n2) + block_system_space<T>(n2, n3);
}

template <typename T>
size_t equilibrate_tp_space(int nspecies, int nreaction) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };
  bump(alignof(T), nreaction * sizeof(T));              // logsvp
  bump(alignof(T), nreaction * nspecies * sizeof(T));   // weight
  bump(alignof(T), nreaction * sizeof(T));              // rhs
  bump(alignof(T), nspecies * nreaction * sizeof(T));   // stoich_active
  bump(alignof(T), nreaction * sizeof(T));              // stoich_sum
  bump(alignof(T), nspecies * sizeof(T));               // xfrac0
  bump(alignof(T), nreaction * nreaction * sizeof(T));  // gain_cpy
  return bytes + leastsq_kkt_space<T>(nreaction, nspecies);
}

template <typename T>
size_t equilibrate_uv_space(int nspecies, int nreaction) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };
  bump(alignof(T), nspecies * sizeof(T));               // intEng
  bump(alignof(T), nspecies * sizeof(T));               // intEng_ddT
  bump(alignof(T), nreaction * sizeof(T));              // logsvp
  bump(alignof(T), nreaction * sizeof(T));              // logsvp_ddT
  bump(alignof(T), nreaction * nspecies * sizeof(T));   // weight
  bump(alignof(T), nreaction * sizeof(T));              // rhs
  bump(alignof(T), nspecies * nreaction * sizeof(T));   // stoich_active
  bump(alignof(T), nreaction * nreaction * sizeof(T));  // gain_cpy
  return bytes + leastsq_kkt_space<T>(nreaction, nspecies);
}

}  // namespace kintera
