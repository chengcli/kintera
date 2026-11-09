// torch
#include <torch/extension.h>

// kintera
#include <kintera/diffusion/diffusion.hpp>

namespace py = pybind11;

void bind_diffusion(py::module& m) {
  m.def(
      "diffusion_tendency",
      [](torch::Tensor y, torch::Tensor Kzz, torch::Tensor dzi,
         std::optional<torch::Tensor> x1f) {
        return kintera::diffusion_tendency(y, Kzz, dzi,
                                           x1f.value_or(torch::Tensor()));
      },
      py::arg("y"), py::arg("Kzz"), py::arg("dzi"),
      py::arg("x1f") = py::none());

  m.def(
      "diffusion_coefficients",
      [](torch::Tensor y, torch::Tensor Kzz, torch::Tensor dzi,
         std::optional<torch::Tensor> x1f) {
        return kintera::diffusion_coefficients(y, Kzz, dzi,
                                               x1f.value_or(torch::Tensor()));
      },
      py::arg("y"), py::arg("Kzz"), py::arg("dzi"),
      py::arg("x1f") = py::none());
}
