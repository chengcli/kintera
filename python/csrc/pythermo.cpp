// torch
#include <torch/extension.h>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_thermo(py::module &m) {
  auto pyThermoOptions = py::class_<kintera::ThermoOptions>(m, "ThermoOptions");

  pyThermoOptions
    .def(py::init<>(),
}
