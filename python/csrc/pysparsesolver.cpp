// pybind11
#include <pybind11/pybind11.h>

// torch
#include <c10/util/Exception.h>
#include <torch/extension.h>

// std
#include <mutex>

// system
#include <dlfcn.h>

namespace py = pybind11;

namespace {

using cuda_csr_solve_fn = torch::Tensor (*)(const torch::Tensor&,
                                            const torch::Tensor&,
                                            const torch::Tensor&,
                                            const torch::Tensor&, double, int);

cuda_csr_solve_fn resolve_cuda_csr_solve_cusolver() {
  static std::once_flag once;
  static cuda_csr_solve_fn cached = nullptr;
  std::call_once(once, []() {
    void* symbol = dlsym(RTLD_DEFAULT, "kintera_cuda_csr_solve_cusolver");
    cached = reinterpret_cast<cuda_csr_solve_fn>(symbol);
  });
  TORCH_CHECK(cached != nullptr,
              "cuda_csr_solve_cusolver is unavailable because kintera was "
              "built without CUDA sparse-solver support");
  return cached;
}

torch::Tensor cuda_csr_solve_cusolver_binding(const torch::Tensor& crow_indices,
                                              const torch::Tensor& col_indices,
                                              const torch::Tensor& values,
                                              const torch::Tensor& rhs,
                                              double tol, int reorder) {
  return resolve_cuda_csr_solve_cusolver()(crow_indices, col_indices, values,
                                           rhs, tol, reorder);
}

}  // namespace

void bind_sparse_solver(py::module& m) {
  m.def("cuda_csr_solve_cusolver", &cuda_csr_solve_cusolver_binding,
        py::arg("crow_indices"), py::arg("col_indices"), py::arg("values"),
        py::arg("rhs"), py::arg("tol") = 0.0, py::arg("reorder") = 0);
}
