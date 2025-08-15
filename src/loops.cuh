#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

// mempool
#include <kintera/mempool/poolalloc.cuh>

namespace kintera {
namespace native {

template <int Arity, typename func_t>
void gpu_kernel(at::TensorIterator& iter, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  at::native::launch_legacy_kernel<128, 1>(numel,
      [=] __device__(int idx) {
      auto offsets = offset_calc.get(idx);
      f(data.data(), offsets.data());
    });
}

template <int Arity, typename func_t>
void pool_kernel(at::TensorIterator& iter, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  void *poolPtr = allocatePools(numel);

  at::native::launch_legacy_kernel<128, 1>(numel,
      [=] __device__(int idx) {
      poolinit(poolPtr, idx);
      auto offsets = offset_calc.get(idx);
      f(data.data(), offsets.data());
    });
  printf("kernel launched with %d threads\n", numel);

  cudaDeviceSynchronize();

  freePools(poolPtr);
  printf("kernel finished\n");
}

}  // namespace native
}  // namespace kintera
