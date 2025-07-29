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
      .ADD_OPTION(double, kintera::ArrheniusOptions, Tref,
                  R"(Reference temperature for the rate constant)")
      .ADD_OPTION(std::vector<kintera::Reaction>, kintera::ArrheniusOptions,
                  reactions,
                  R"(Reactions for which the rate constants are defined)")
      .ADD_OPTION(
          std::vector<double>, kintera::ArrheniusOptions, A,
          R"(Pre-exponential factor for the rate constant, in units of `units`)")
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptions, b,
                  R"(Dimensionless temperature exponent for the rate constant)")
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptions, Ea_R,
                  R"(Activation energy in K for the rate constant)")
      .ADD_OPTION(
          std::vector<double>, kintera::ArrheniusOptions, E4_R,
          R"(Additional 4th parameter in the rate expression for the rate constant)");

  ADD_KINTERA_MODULE(Arrhenius, ArrheniusOptions,
                     R"(Arrhenius rate kinetics model)", py::arg("temp"),
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
      .ADD_OPTION(
          double, kintera::EvaporationOptions, Tref,
          R"(Reference temperature [K] for the evaporation rate constant)")
      .ADD_OPTION(
          double, kintera::EvaporationOptions, Pref,
          R"(Reference pressure [pa] for the evaporation rate constant)")
      .ADD_OPTION(
          std::vector<double>, kintera::EvaporationOptions, diff_c,
          R"(Diffusivity [m^2/s] at reference temperature and pressure)")
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, diff_T,
                  R"(Diffusivity temperature exponent)")
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, diff_P,
                  R"(Diffusivity pressure exponent)")
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, vm,
                  R"(Molar volume [m^3/mol])")
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptions, diameter,
                  R"(Particle diameter [m])");

  ADD_KINTERA_MODULE(Evaporation, EvaporationOptions,
                     R"(Evaporation rate kinetics model)", py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"));

  ////////////// Kinetics //////////////
  auto pyKineticsOptions =
      py::class_<kintera::KineticsOptions, kintera::SpeciesThermo>(
          m, "KineticsOptions");

  pyKineticsOptions.def(py::init<>())
      .def_static("from_yaml", &kintera::KineticsOptions::from_yaml,
                  py::arg("filename"))
      .def("reactions", &kintera::KineticsOptions::reactions)
      .ADD_OPTION(double, kintera::KineticsOptions, Tref,
                  R"(Reference temperature [K] for the kinetics reactions)")
      .ADD_OPTION(double, kintera::KineticsOptions, Pref,
                  R"(Reference pressure [pa] for the kinetics reactions)")
      .ADD_OPTION(kintera::ArrheniusOptions, kintera::KineticsOptions,
                  arrhenius, R"(Options for Arrhenius rate constants)")
      .ADD_OPTION(kintera::CoagulationOptions, kintera::KineticsOptions,
                  coagulation, R"(Options for coagulation reactions)")
      .ADD_OPTION(kintera::EvaporationOptions, kintera::KineticsOptions,
                  evaporation, R"(Options for evaporation reactions)")
      .ADD_OPTION(
          bool, kintera::KineticsOptions, evolve_temperature,
          R"(Whether to evolve temperature during kinetics calculation, default is false)");

  ADD_KINTERA_MODULE(Kinetics, KineticsOptions, R"(Evolve kinetic reactions)",
                     py::arg("temp"), py::arg("conc"), py::arg("conc"))
      .def("jacobian", &kintera::KineticsImpl::jacobian, py::arg("temp"),
           py::arg("conc"), py::arg("cvol"), py::arg("rate"), py::arg("rc_ddC"),
           py::arg("rc_ddT") = torch::nullopt);
}
