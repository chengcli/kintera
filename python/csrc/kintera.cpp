// torch
#include <torch/extension.h>

// fmt
#include <fmt/format.h>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pydisort, m) {
  m.attr("__name__") = "disort";
  m.doc() = R"(Photochemistry model)";
}
