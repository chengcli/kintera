// torch
#include <torch/extension.h>

// kintera
#include <kintera/diffusion/diffusion.hpp>

namespace py = pybind11;

void bind_diffusion(py::module &m) {
  m.def("diffusion_tendency", &kintera::diffusion_tendency, py::arg("y"),
        py::arg("Kzz"), py::arg("dzi"));

  m.def("diffusion_coefficients", &kintera::diffusion_coefficients,
        py::arg("y"), py::arg("Kzz"), py::arg("dzi"));
}
