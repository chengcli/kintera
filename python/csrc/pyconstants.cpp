// torch
#include <torch/extension.h>

// kintera
#include <kintera/constants.h>

namespace py = pybind11;
using namespace kintera;

void bind_constants(py::module &m) {
  py::module_ c = m.def_submodule("constants", "Physical constants");

  c.attr("Rgas") = py::float_(constants::Rgas);
  c.attr("Avogadro") = py::float_(constants::Avogadro);
}
