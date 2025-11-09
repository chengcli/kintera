// torch
#include <torch/extension.h>

// pybind11
#include <pybind11/stl.h>

// kintera
#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_kinetics(py::module &m) {
  ////////////// Arrhenius //////////////
  auto pyArrheniusOptions =
      py::class_<kintera::ArrheniusOptions>(m, "ArrheniusOptions");

  pyArrheniusOptions.def(py::init<>())
      .ADD_OPTION(double, kintera::ArrheniusOptions, Tref)
      .ADD_OPTION(std::vector<kintera::Reaction>, kintera::ArrheniusOptions,
                  reactions)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptions, A)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptions, b)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptions, Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptions, E4_R);

  ADD_KINTERA_MODULE(Arrhenius, ArrheniusOptions, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"));

  ////////////// Coagulation //////////////
  auto pyCoagulationOptions =
      py::class_<kintera::CoagulationOptions, kintera::ArrheniusOptions>(
          m, "CoagulationOptions");

  pyCoagulationOptions.def(py::init<>());

  ///////////// Evaporation //////////////
  auto pyEvaporationOptions =
      py::class_<kintera::EvaporationOptions, kintera::NucleationOptions>(
          m, "EvaporationOptions");

  pyEvaporationOptions.def(py::init<>())
      .ADD_OPTION(double, kintera::EvaporationOptions, Tref)
      .ADD_OPTION(double, kintera::EvaporationOptions, Pref)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, diff_c)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, diff_T)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, diff_P)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, vm)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, diameter);

  ADD_KINTERA_MODULE(Evaporation, EvaporationOptions, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"));

  ////////////// Kinetics //////////////
  auto pyKineticsOptions =
      py::class_<kintera::KineticsOptions, kintera::SpeciesThermo>(
          m, "KineticsOptions");

  pyKineticsOptions.def(py::init<>())
      .def_static("from_yaml", &kintera::KineticsOptions::from_yaml,
                  py::arg("filename"))
      .def("reactions", &kintera::KineticsOptions::reactions)
      .ADD_OPTION(double, kintera::KineticsOptions, Tref)
      .ADD_OPTION(double, kintera::KineticsOptions, Pref)
      .ADD_OPTION(kintera::ArrheniusOptions, kintera::KineticsOptions,
                  arrhenius)
      .ADD_OPTION(kintera::CoagulationOptions, kintera::KineticsOptions,
                  coagulation)
      .ADD_OPTION(kintera::EvaporationOptions, kintera::KineticsOptions,
                  evaporation)
      .ADD_OPTION(bool, kintera::KineticsOptions, evolve_temperature);

  ADD_KINTERA_MODULE(Kinetics, KineticsOptions, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"))
      .def("forward_nogil",
           [](kintera::KineticsImpl &self, torch::Tensor temp,
              torch::Tensor pres, torch::Tensor conc) {
             py::gil_scoped_release no_gil;
             return self.forward(temp, pres, conc);
           })
      .def("jacobian", &kintera::KineticsImpl::jacobian, py::arg("temp"),
           py::arg("conc"), py::arg("cvol"), py::arg("rate"), py::arg("rc_ddC"),
           py::arg("rc_ddT") = torch::nullopt);
}
