#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

// torch
#include <ATen/Functions.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAException.h>

namespace kintera {
namespace native {

inline size_t get_max_dynamic_shared_memory(int device) {
  static std::mutex mutex;
  static std::vector<size_t> max_dynamic_smem_by_device;

  std::lock_guard<std::mutex> guard(mutex);
  if (device >= static_cast<int>(max_dynamic_smem_by_device.size())) {
    max_dynamic_smem_by_device.resize(device + 1, 0);
  }

  auto &cached = max_dynamic_smem_by_device[device];
  if (cached != 0) {
    return cached;
  }

  // Query max allowed per-block shared memory once per device.  Some runtimes
  // report sharedMemPerBlockOptin as 1 even when opt-in dynamic shared memory
  // is available, so prefer the CUDA device attribute and never let a bad
  // opt-in value reduce the ordinary per-block limit.
  auto *prop = at::cuda::getDeviceProperties(device);
  cached = prop->sharedMemPerBlock;

  int attr_optin_smem = 0;
  cudaError_t attr_err = cudaDeviceGetAttribute(
      &attr_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (attr_err == cudaSuccess) {
    cached = std::max(cached, static_cast<size_t>(attr_optin_smem));
  } else {
    C10_CUDA_CLEAR_ERROR();
    cached =
        std::max(cached, static_cast<size_t>(prop->sharedMemPerBlockOptin));
  }

  return cached;
}

template <typename func_t>
__global__ void element_kernel(int64_t numel, func_t f) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Shared memory allocation
  extern __shared__ unsigned char memory[];
  char *smem = reinterpret_cast<char *>(memory);

  if (idx < numel) {
    f(idx, smem);
  }
}

template <int Arity, typename func_t>
void gpu_kernel(at::TensorIterator &iter, const func_t &f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char *, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char *>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  at::native::launch_legacy_kernel<128, 1>(numel, [=] __device__(int idx) {
    auto offsets = offset_calc.get(idx);
    f(data.data(), offsets.data());
  });
}

template <int Threads, int Arity, typename func_t>
void gpu_mem_kernel(at::TensorIterator &iter, int work_size, const func_t &f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char *, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char *>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  dim3 block(Threads);
  dim3 grid((numel + block.x - 1) / block.x);
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t shared = block.x * work_size;

  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  auto *prop = at::cuda::getDeviceProperties(device);
  size_t max_dynamic_smem = get_max_dynamic_shared_memory(device);
  // printf("max_dynamic_smem = %zu\n", max_dynamic_smem);

  auto device_lambda = [=] __device__(int idx, char *smem) {
    auto offsets = offset_calc.get(idx);
    int tid = threadIdx.x;
    f(data.data(), offsets.data(), smem + tid * work_size);
  };

  // request the full size
  auto kernelPtr = element_kernel<decltype(device_lambda)>;
  if (shared > (size_t)max_dynamic_smem) {
    TORCH_CHECK(false, "Requested shared memory (", shared,
                " bytes) exceeds device maximum (", max_dynamic_smem,
                " bytes).");
  }
  if (shared > prop->sharedMemPerBlock) {
    static std::mutex attr_mutex;
    static std::vector<std::unique_ptr<std::once_flag>> attr_once_by_device;

    std::once_flag *attr_once = nullptr;
    {
      std::lock_guard<std::mutex> guard(attr_mutex);
      if (device >= static_cast<int>(attr_once_by_device.size())) {
        attr_once_by_device.resize(device + 1);
      }
      auto &flag = attr_once_by_device[device];
      if (!flag) {
        flag = std::make_unique<std::once_flag>();
      }
      attr_once = flag.get();
    }

    std::call_once(*attr_once, [=] {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernelPtr, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(max_dynamic_smem)));
    });
  }

  /*std::cout << "block = " << block.x
            << ", grid = " << grid.x
            << ", shared = " << shared
            << ", work_size = " << work_size
            << std::endl;*/

  element_kernel<<<grid, block, shared, stream>>>(numel, device_lambda);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
__global__ void global_mem_element_kernel(int64_t numel, char *workspace,
                                          size_t work_size, func_t f) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    f(idx, workspace + idx * work_size);
  }
}

template <int Threads, int Arity, typename func_t>
void gpu_global_mem_kernel(at::TensorIterator &iter, size_t work_size,
                           const func_t &f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);
  TORCH_CHECK(work_size > 0, "CUDA workspace size must be positive");

  std::array<char *, Arity> data;
  for (int i = 0; i < Arity; ++i) {
    data[i] = reinterpret_cast<char *>(iter.data_ptr(i));
  }

  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }
  TORCH_CHECK(work_size <=
                  static_cast<size_t>(std::numeric_limits<int64_t>::max()),
              "CUDA workspace per cell is too large");
  auto workspace =
      at::empty({numel, static_cast<int64_t>(work_size)},
                at::TensorOptions().device(iter.device()).dtype(at::kByte));
  auto *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
  auto offset_calc = ::make_offset_calculator<Arity>(iter);

  auto device_lambda = [=] __device__(int64_t idx, char *work) {
    auto offsets = offset_calc.get(idx);
    f(data.data(), offsets.data(), work);
  };

  dim3 block(Threads);
  dim3 grid((numel + block.x - 1) / block.x);
  auto stream = at::cuda::getCurrentCUDAStream();
  global_mem_element_kernel<<<grid, block, 0, stream>>>(
      numel, workspace_ptr, work_size, device_lambda);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace native
} // namespace kintera
