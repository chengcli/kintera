// torch
#include <torch/extension.h>

// pybind11
#include <pybind11/stl.h>

// kintera
#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/kinetics/falloff.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_kinetics(py::module& m) {
  ////////////// Arrhenius //////////////
  auto pyArrheniusOptions =
      py::class_<kintera::ArrheniusOptionsImpl, kintera::ArrheniusOptions>(
          m, "ArrheniusOptions");

  pyArrheniusOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::ArrheniusOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("ArrheniusOptions({})", ss.str());
           })
      .ADD_OPTION(double, kintera::ArrheniusOptionsImpl, Tref)
      .ADD_OPTION(std::vector<kintera::Reaction>, kintera::ArrheniusOptionsImpl,
                  reactions)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptionsImpl, A)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptionsImpl, b)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptionsImpl, Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::ArrheniusOptionsImpl, E4_R);

  ADD_KINTERA_MODULE(Arrhenius, ArrheniusOptions, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"));

  ////////////// Coagulation //////////////
  auto pyCoagulationOptions =
      py::class_<kintera::CoagulationOptionsImpl, kintera::ArrheniusOptionsImpl,
                 kintera::CoagulationOptions>(m, "CoagulationOptions");

  pyCoagulationOptions.def(py::init<>())
      .def("__repr__", [](const kintera::CoagulationOptions& self) {
        std::stringstream ss;
        self->report(ss);
        return fmt::format("CoagulationOptions({})", ss.str());
      });

  ///////////// Evaporation //////////////
  auto pyEvaporationOptions =
      py::class_<kintera::EvaporationOptionsImpl,
                 kintera::NucleationOptionsImpl, kintera::EvaporationOptions>(
          m, "EvaporationOptions");

  pyEvaporationOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::EvaporationOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("EvaporationOptions({})", ss.str());
           })
      .ADD_OPTION(double, kintera::EvaporationOptionsImpl, Tref)
      .ADD_OPTION(double, kintera::EvaporationOptionsImpl, Pref)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptionsImpl, diff_c)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptionsImpl, diff_T)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptionsImpl, diff_P)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptionsImpl, vm)
      .ADD_OPTION(std::vector<double>, kintera::EvaporationOptionsImpl,
                  diameter);

  ADD_KINTERA_MODULE(Evaporation, EvaporationOptions, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"));

  ////////////// Falloff //////////////
  auto pyFalloffOptions =
      py::class_<kintera::FalloffOptionsImpl, kintera::FalloffOptions>(
          m, "FalloffOptions");

  pyFalloffOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::FalloffOptions &self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("FalloffOptions({})", ss.str());
           })
      .ADD_OPTION(double, kintera::FalloffOptionsImpl, Tref)
      .ADD_OPTION(std::string, kintera::FalloffOptionsImpl, units)
      .ADD_OPTION(std::vector<kintera::Reaction>,
                  kintera::FalloffOptionsImpl, reactions)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, k0_A)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, k0_b)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, k0_Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, kinf_A)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, kinf_b)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, kinf_Ea_R)
      .ADD_OPTION(std::vector<int>, kintera::FalloffOptionsImpl, falloff_types)
      .ADD_OPTION(std::vector<bool>, kintera::FalloffOptionsImpl, is_three_body)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, troe_A)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, troe_T3)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, troe_T1)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, troe_T2)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, sri_A)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, sri_B)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, sri_C)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, sri_D)
      .ADD_OPTION(std::vector<double>, kintera::FalloffOptionsImpl, sri_E)
      .ADD_OPTION(std::vector<kintera::Composition>,
                  kintera::FalloffOptionsImpl, efficiencies);

  ADD_KINTERA_MODULE(Falloff, FalloffOptions, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"))
      .def("pretty_print", &kintera::FalloffImpl::pretty_print);

  ////////////// Kinetics //////////////
  auto pyKineticsOptions =
      py::class_<kintera::KineticsOptionsImpl, kintera::SpeciesThermoImpl,
                 kintera::KineticsOptions>(m, "KineticsOptions");

  pyKineticsOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::KineticsOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("KineticsOptions({})", ss.str());
           })
      .def_static("from_yaml",
                  py::overload_cast<std::string const&, bool>(
                      &kintera::KineticsOptionsImpl::from_yaml),
                  py::arg("filename"), py::arg("verbose") = false)
      .def("reactions", &kintera::KineticsOptionsImpl::reactions)
      .ADD_OPTION(double, kintera::KineticsOptionsImpl, Tref)
      .ADD_OPTION(double, kintera::KineticsOptionsImpl, Pref)
      .ADD_OPTION(kintera::ArrheniusOptions, kintera::KineticsOptionsImpl,
                  arrhenius)
      .ADD_OPTION(kintera::CoagulationOptions, kintera::KineticsOptionsImpl,
                  coagulation)
      .ADD_OPTION(kintera::EvaporationOptions, kintera::KineticsOptionsImpl,
                  evaporation)
      .ADD_OPTION(kintera::FalloffOptions, kintera::KineticsOptionsImpl,
                  falloff)
      .ADD_OPTION(bool, kintera::KineticsOptionsImpl, evolve_temperature);

  ADD_KINTERA_MODULE(Kinetics, KineticsOptions, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"))
      .def("forward_nogil",
           [](kintera::KineticsImpl& self, torch::Tensor temp,
              torch::Tensor pres, torch::Tensor conc) {
             py::gil_scoped_release no_gil;
             return self.forward(temp, pres, conc);
           })
      .def("jacobian", &kintera::KineticsImpl::jacobian, py::arg("temp"),
           py::arg("conc"), py::arg("cvol"), py::arg("rate"), py::arg("rc_ddC"),
           py::arg("rc_ddT") = torch::nullopt);
}
