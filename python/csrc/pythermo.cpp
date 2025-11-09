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
      py::class_<kintera::NucleationOptions>(m, "NucleationOptions");

  pyNucleationOptions
      .def(py::init<>())

      .def("__repr__",
           [](const kintera::NucleationOptions &self) {
             std::stringstream ss;
             self.report(ss);
             return fmt::format("NucleationOptions({})", ss.str());
           })

      .ADD_OPTION(std::vector<double>, kintera::NucleationOptions, minT, "")

      .ADD_OPTION(std::vector<double>, kintera::NucleationOptions, maxT, "")

      .ADD_OPTION(std::vector<std::string>, kintera::NucleationOptions, logsvp, "")

      .ADD_OPTION(std::vector<kintera::Reaction>, kintera::NucleationOptions,
                  reactions, "");

  auto pyThermoOptions =
      py::class_<kintera::ThermoOptions, kintera::SpeciesThermo>(
          m, "ThermoOptions");

  pyThermoOptions
      .def(py::init<>())

      .def("__repr__",
           [](const kintera::ThermoOptions &self) {
             std::stringstream ss;
             self.report(ss);
             return fmt::format("ThermoOptions({})", ss.str());
           })

      .def("reactions", &kintera::ThermoOptions::reactions)

      .def("from_yaml",
           py::overload_cast<std::string const &>(
               &kintera::ThermoOptions::from_yaml))

      .ADD_OPTION(double, kintera::ThermoOptions, Tref, "")

      .ADD_OPTION(double, kintera::ThermoOptions, Pref, "")

      .ADD_OPTION(kintera::NucleationOptions, kintera::ThermoOptions,
                  nucleation, "")

      .ADD_OPTION(int, kintera::ThermoOptions, max_iter, "")

      .ADD_OPTION(double, kintera::ThermoOptions, ftol, "");

  ADD_KINTERA_MODULE(ThermoY, ThermoOptions, "",
                     py::arg("rho"), py::arg("intEng"), py::arg("yfrac"),
                     py::arg("warm_start") = false,
                     py::arg("diag") = py::none())

      .def("compute", &kintera::ThermoYImpl::compute,
           py::arg("ab"), py::arg("args"));

  ADD_KINTERA_MODULE(ThermoX, ThermoOptions, "",
                     py::arg("temp"), py::arg("pres"), py::arg("xfrac"),
                     py::arg("warm_start") = false,
                     py::arg("diag") = py::none())

      .def("compute", &kintera::ThermoXImpl::compute,
           py::arg("ab"), py::arg("args"))

      .def("effective_cp", &kintera::ThermoXImpl::effective_cp,
           py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("gain"),
           py::arg("conc") = py::none())

      .def(
          "extrapolate_ad",
          py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, double,
                            bool>(&kintera::ThermoXImpl::extrapolate_ad),
          py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("dlnp"),
          py::arg("verbose") = false)

      .def("extrapolate_ad",
           py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor,
                             double, double, bool>(
               &kintera::ThermoXImpl::extrapolate_ad),
           py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("grav"),
           py::arg("dz"), py::arg("verbose") = false);

  m.def("relative_humidity", &kintera::relative_humidity,
        py::arg("temp"), py::arg("conc"), py::arg("stoich"), py::arg("op"));
}
