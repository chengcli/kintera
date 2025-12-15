// torch
#include <torch/extension.h>

// pybind11
#include <pybind11/stl.h>

// kintera
#include <kintera/thermo/relative_humidity.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_thermo(py::module &m) {
  auto pyNucleationOptions =
      py::class_<kintera::NucleationOptionsImpl, kintera::NucleationOptions>(
          m, "NucleationOptions");

  pyNucleationOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::NucleationOptions &self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("NucleationOptions({})", ss.str());
           })
      .ADD_OPTION(std::vector<double>, kintera::NucleationOptionsImpl, minT)
      .ADD_OPTION(std::vector<double>, kintera::NucleationOptionsImpl, maxT)
      .ADD_OPTION(std::vector<std::string>, kintera::NucleationOptionsImpl,
                  logsvp)
      .ADD_OPTION(std::vector<kintera::Reaction>,
                  kintera::NucleationOptionsImpl, reactions);

  auto pyThermoOptions =
      py::class_<kintera::ThermoOptionsImpl, kintera::SpeciesThermoImpl,
                 kintera::ThermoOptions>(m, "ThermoOptions");

  pyThermoOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::ThermoOptions &self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("ThermoOptions({})", ss.str());
           })
      .def_static("from_yaml",
                  py::overload_cast<std::string const &, bool>(
                      &kintera::ThermoOptionsImpl::from_yaml),
                  py::arg("filename"), py::arg("verbose") = false)
      .def("reactions", &kintera::ThermoOptionsImpl::reactions)
      .ADD_OPTION(double, kintera::ThermoOptionsImpl, Tref)
      .ADD_OPTION(double, kintera::ThermoOptionsImpl, Pref)
      .ADD_OPTION(kintera::NucleationOptions, kintera::ThermoOptionsImpl,
                  nucleation)
      .ADD_OPTION(int, kintera::ThermoOptionsImpl, max_iter)
      .ADD_OPTION(double, kintera::ThermoOptionsImpl, ftol);

  ADD_KINTERA_MODULE(ThermoY, ThermoOptions, py::arg("rho"), py::arg("intEng"),
                     py::arg("yfrac"), py::arg("warm_start") = false,
                     py::arg("diag") = py::none())

      .def("compute", &kintera::ThermoYImpl::compute, py::arg("ab"),
           py::arg("args"));

  ADD_KINTERA_MODULE(ThermoX, ThermoOptions, py::arg("temp"), py::arg("pres"),
                     py::arg("xfrac"), py::arg("warm_start") = false,
                     py::arg("diag") = py::none())

      .def("compute", &kintera::ThermoXImpl::compute, py::arg("ab"),
           py::arg("args"))
      .def("effective_cp", &kintera::ThermoXImpl::effective_cp, py::arg("temp"),
           py::arg("pres"), py::arg("xfrac"), py::arg("gain"),
           py::arg("conc") = py::none())
      .def("extrapolate_ad",
           py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor,
                             double, double, bool>(
               &kintera::ThermoXImpl::extrapolate_ad),
           py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("dlnp"),
           py::arg("ds_dlnp") = 0., py::arg("verbose") = false)

      .def("extrapolate_ad",
           py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor,
                             double, double, double, bool>(
               &kintera::ThermoXImpl::extrapolate_ad),
           py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("grav"),
           py::arg("dz"), py::arg("ds_dlnp") = 0., py::arg("verbose") = false);

  m.def("relative_humidity", &kintera::relative_humidity, py::arg("temp"),
        py::arg("conc"), py::arg("stoich"), py::arg("op"));
}
