// pybind11
#include <pybind11/pybind11.h>

// torch
#include <torch/extension.h>

// kintera
#include <src/math/sparse_solver.hpp>

namespace py = pybind11;

void bind_sparse_solver(py::module& m) {
  m.def("cuda_csr_solve_cusolver", &kintera::cuda_csr_solve_cusolver,
        py::arg("crow_indices"), py::arg("col_indices"), py::arg("values"),
        py::arg("rhs"), py::arg("tol") = 0.0, py::arg("reorder") = 0);
}
