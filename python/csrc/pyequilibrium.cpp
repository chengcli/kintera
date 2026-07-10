#include <sstream>

#include <fmt/format.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <kintera/equilibrium/equilibrium.hpp>

#include "pyoptions.hpp"

namespace py = pybind11;

void bind_equilibrium(py::module &m) {
  auto pyOptions =
      py::class_<kintera::EquilibriumOptionsImpl, kintera::EquilibriumOptions>(
          m, "EquilibriumOptions");

  pyOptions.def(py::init<>())
      .def("__repr__",
           [](kintera::EquilibriumOptions const &self) {
             std::stringstream ss;
             self->report(ss);
             return ss.str();
           })
      .def_static("from_yaml", &kintera::EquilibriumOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verbose") = false)
      .ADD_OPTION(std::vector<std::string>, kintera::EquilibriumOptionsImpl,
                  components)
      .ADD_OPTION(std::vector<std::string>, kintera::EquilibriumOptionsImpl,
                  elements)
      .ADD_OPTION(std::vector<std::string>, kintera::EquilibriumOptionsImpl,
                  phases)
      .ADD_OPTION(std::vector<std::string>, kintera::EquilibriumOptionsImpl,
                  reactions)
      .ADD_OPTION(std::vector<int>, kintera::EquilibriumOptionsImpl, phase_ids)
      .ADD_OPTION(kintera::Matrix, kintera::EquilibriumOptionsImpl, stoich)
      .ADD_OPTION(kintera::Matrix, kintera::EquilibriumOptionsImpl,
                  element_matrix)
      .ADD_OPTION(int, kintera::EquilibriumOptionsImpl, gas_phase)
      .ADD_OPTION(double, kintera::EquilibriumOptionsImpl, standard_pressure)
      .ADD_OPTION(int, kintera::EquilibriumOptionsImpl, max_iter)
      .ADD_OPTION(double, kintera::EquilibriumOptionsImpl, ftol)
      .ADD_OPTION(double, kintera::EquilibriumOptionsImpl, mole_floor)
      .def("validate", &kintera::EquilibriumOptionsImpl::validate);

  ADD_KINTERA_MODULE(Equilibrium, EquilibriumOptions,
                     &kintera::EquilibriumImpl::forward, py::arg("temp"),
                     py::arg("pres"), py::arg("moles"), py::arg("log_k"),
                     py::arg("warm_start") = false);
}
