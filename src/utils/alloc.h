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

enum class PoolBackend { Shared, DisortGlobal };

namespace shared_pool {
struct State {
  char* base;
  uint32_t capacity;
  uint32_t offset;
};

#ifdef __CUDACC__
static __device__ inline State& state() {
  extern __shared__ unsigned char memory[];
  return reinterpret_cast<State*>(memory)[threadIdx.x];
}
#endif
}  // namespace shared_pool

DISPATCH_MACRO inline void shared_pool_init(char* base, size_t capacity) {
#ifdef __CUDA_ARCH__
  if (capacity > UINT32_MAX) pmem::trap("shared pool too large", capacity);
  shared_pool::state() = {base, static_cast<uint32_t>(capacity), 0};
#endif
}

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline size_t pool_mark() {
#ifdef __CUDA_ARCH__
  if constexpr (Backend == PoolBackend::Shared) {
    return shared_pool::state().offset;
  } else {
    return pmem::pool_state(pmem::slice_base())->offset;
  }
#else
  return 0;
#endif
}

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline void pool_rewind(size_t offset) {
#ifdef __CUDA_ARCH__
  if constexpr (Backend == PoolBackend::Shared) {
    shared_pool::state().offset = static_cast<uint32_t>(offset);
  } else {
    pmem::pool_state(pmem::slice_base())->offset =
        static_cast<uint32_t>(offset);
  }
#endif
}

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline void* pmalloc(size_t bytes) {
#ifdef __CUDA_ARCH__
  if constexpr (Backend == PoolBackend::Shared) {
    auto& state = shared_pool::state();
    size_t size = align_up(bytes == 0 ? 8 : bytes, 8);
    if (size > state.capacity - state.offset)
      pmem::trap("shared pool exhausted", bytes);
    void* result = state.base + state.offset;
    state.offset += static_cast<uint32_t>(size);
    return result;
  }
#endif
  return ::pmalloc(bytes);
}

template <PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO inline void pfree(void* ptr) {
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
  return bytes + leastsq_kkt_space<T>(nreaction, nspecies);
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
  return bytes + leastsq_kkt_space<T>(nreaction, nspecies);
}

}  // namespace kintera
