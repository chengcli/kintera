#pragma once

// C/C++
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// base
#include <cdisort213/pmem.h>
#include <configure.h>

namespace kintera {

DISPATCH_MACRO inline uintptr_t align_up(uintptr_t p, size_t a) {
  // a must be power of two; works for 4, 8, 16, ...
  return (p + (a - 1)) & ~(a - 1);
}

enum class PoolBackend { Shared, DisortGlobal, HostBump };

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline size_t pool_mark(char* work) {
  if constexpr (Backend == PoolBackend::HostBump)
    return reinterpret_cast<uintptr_t>(work);
#ifdef __CUDA_ARCH__
  if constexpr (Backend == PoolBackend::Shared) {
    return reinterpret_cast<uintptr_t>(work);
  } else {
    return pmem::pool_state(pmem::slice_base())->offset;
  }
#else
  return 0;
#endif
}

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline void pool_rewind(char*& work, size_t mark) {
  if constexpr (Backend == PoolBackend::HostBump)
    work = reinterpret_cast<char*>(mark);
#ifdef __CUDA_ARCH__
  if constexpr (Backend == PoolBackend::Shared) {
    work = reinterpret_cast<char*>(mark);
  } else {
    pmem::pool_state(pmem::slice_base())->offset = static_cast<uint32_t>(mark);
  }
#endif
}

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline void* pmalloc(char*& work, size_t bytes) {
  if constexpr (Backend == PoolBackend::HostBump) {
    void* result = work;
    work += align_up(bytes == 0 ? 8 : bytes, 8);
    return result;
  }
#ifdef __CUDA_ARCH__
  if constexpr (Backend == PoolBackend::Shared) {
    void* result = work;
    work += align_up(bytes == 0 ? 8 : bytes, 8);
    return result;
  }
#endif
  return ::pmalloc(bytes);
}

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline void pfree(void* ptr) {
  if constexpr (Backend == PoolBackend::HostBump) return;
#ifdef __CUDA_ARCH__
  if constexpr (Backend == PoolBackend::Shared) return;
#endif
  ::pfree(ptr);
}

inline size_t pool_allocation_bytes(size_t bytes) {
  return static_cast<size_t>(align_up(bytes == 0 ? 8 : bytes, 8));
}

inline size_t pool_workspace_bytes(size_t bytes) {
  return pool_allocation_bytes(bytes);
}

template <typename T>
size_t ludcmp_space(int n) {
  size_t bytes = 0;
  auto bump = [&](size_t, size_t nbytes) {
    bytes += pool_allocation_bytes(nbytes);
  };
  bump(alignof(T), n * sizeof(T));  // vv
  return bytes;
}

template <typename T>
size_t psolve_space(int n) {
  size_t bytes = 0;
  auto bump = [&](size_t, size_t nbytes) {
    bytes += pool_allocation_bytes(nbytes);
  };

  bump(alignof(T), n * n * sizeof(T));  // ATA
  bump(alignof(T), n * n * sizeof(T));  // V
  bump(alignof(T), n * sizeof(T));      // eval
  bump(alignof(T), n * sizeof(T));      // vi
  bump(alignof(T), n * sizeof(T));      // Avi
  bump(alignof(T), n * sizeof(T));      // b0
  return bytes;
}

template <typename T>
size_t leastsq_kkt_space(int n2, int n3) {
  int size = n2 + n3;

  size_t bytes = 0;
  auto bump = [&](size_t, size_t nbytes) {
    bytes += pool_allocation_bytes(nbytes);
  };
  bump(alignof(T), size * size * sizeof(T));  // aug
  bump(alignof(T), n2 * n2 * sizeof(T));      // ata
  bump(alignof(T), n2 * sizeof(T));           // column_norm
  bump(alignof(T), size * sizeof(T));         // rhs
  bump(alignof(T), n3 * sizeof(T));           // eval
  bump(alignof(int), n3 * sizeof(int));       // ct_indx
  bump(alignof(int), size * sizeof(int));     // lu_indx
  bump(alignof(int), size * sizeof(int));     // skip_row
  return bytes + ludcmp_space<T>(n2 + n3);
}

template <typename T>
size_t equilibrate_tp_space(int nspecies, int nreaction) {
  size_t bytes = 0;
  auto bump = [&](size_t, size_t nbytes) {
    bytes += pool_allocation_bytes(nbytes);
  };
  bump(alignof(T), nreaction * sizeof(T));              // logsvp
  bump(alignof(T), nreaction * nspecies * sizeof(T));   // weight
  bump(alignof(T), nreaction * sizeof(T));              // rhs
  bump(alignof(T), nspecies * nreaction * sizeof(T));   // stoich_active
  bump(alignof(T), nreaction * sizeof(T));              // stoich_sum
  bump(alignof(T), nspecies * sizeof(T));               // xfrac0
  bump(alignof(T), nreaction * nreaction * sizeof(T));  // gain_cpy
  bump(alignof(T), nspecies * sizeof(T));               // theta
  return bytes + leastsq_kkt_space<T>(nreaction, nspecies);
}

template <typename T>
size_t equilibrate_uv_partition_space(int nspecies, int nreaction) {
  return 2 * pool_allocation_bytes(nspecies * sizeof(T)) +
         pool_allocation_bytes(nreaction * sizeof(T));
}

template <typename T>
size_t equilibrate_uv_space(int nspecies, int nreaction) {
  size_t bytes = 0;
  auto bump = [&](size_t, size_t nbytes) {
    bytes += pool_allocation_bytes(nbytes);
  };
  bump(alignof(T), nspecies * sizeof(T));               // intEng
  bump(alignof(T), nspecies * sizeof(T));               // intEng_ddT
  bump(alignof(T), nreaction * sizeof(T));              // logsvp
  bump(alignof(T), nreaction * sizeof(T));              // logsvp_ddT
  bump(alignof(T), nreaction * nspecies * sizeof(T));   // weight
  bump(alignof(T), nreaction * sizeof(T));              // rhs
  bump(alignof(T), nspecies * nreaction * sizeof(T));   // stoich_active
  bump(alignof(T), nspecies * sizeof(T));               // conc0
  bump(alignof(T), nreaction * nreaction * sizeof(T));  // gain_cpy
  bump(alignof(T), nspecies * sizeof(T));               // theta
  return bytes + leastsq_kkt_space<T>(nreaction, nspecies);
}

}  // namespace kintera
