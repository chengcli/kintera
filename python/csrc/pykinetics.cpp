// torch
#include <torch/extension.h>

// pybind11
#include <pybind11/stl.h>

// kintera
#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/kinetics/lindemann_falloff.hpp>
#include <kintera/kinetics/sri_falloff.hpp>
#include <kintera/kinetics/three_body.hpp>
#include <kintera/kinetics/troe_falloff.hpp>

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

  ADD_KINTERA_MODULE(Arrhenius, ArrheniusOptions,
                     &kintera::ArrheniusImpl::forward, py::arg("temp"),
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

  ADD_KINTERA_MODULE(Evaporation, EvaporationOptions,
                     &kintera::EvaporationImpl::forward, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"));

  ////////////// Three-Body //////////////
  auto pyThreeBodyOptions =
      py::class_<kintera::ThreeBodyOptionsImpl, kintera::ThreeBodyOptions>(
          m, "ThreeBodyOptions");

  pyThreeBodyOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::ThreeBodyOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("ThreeBodyOptions({})", ss.str());
           })
      .ADD_OPTION(double, kintera::ThreeBodyOptionsImpl, Tref)
      .ADD_OPTION(std::string, kintera::ThreeBodyOptionsImpl, units)
      .ADD_OPTION(std::vector<kintera::Reaction>, kintera::ThreeBodyOptionsImpl,
                  reactions)
      .ADD_OPTION(std::vector<double>, kintera::ThreeBodyOptionsImpl, k0_A)
      .ADD_OPTION(std::vector<double>, kintera::ThreeBodyOptionsImpl, k0_b)
      .ADD_OPTION(std::vector<double>, kintera::ThreeBodyOptionsImpl, k0_Ea_R)
      .ADD_OPTION(std::vector<kintera::Composition>,
                  kintera::ThreeBodyOptionsImpl, efficiencies);

  ADD_KINTERA_MODULE(ThreeBody, ThreeBodyOptions,
                     &kintera::ThreeBodyImpl::forward, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"))
      .def("pretty_print", &kintera::ThreeBodyImpl::pretty_print);

  ////////////// Lindemann Falloff //////////////
  auto pyLindemannFalloffOptions =
      py::class_<kintera::LindemannFalloffOptionsImpl,
                 kintera::LindemannFalloffOptions>(m,
                                                   "LindemannFalloffOptions");

  pyLindemannFalloffOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::LindemannFalloffOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("LindemannFalloffOptions({})", ss.str());
           })
      .ADD_OPTION(double, kintera::LindemannFalloffOptionsImpl, Tref)
      .ADD_OPTION(std::string, kintera::LindemannFalloffOptionsImpl, units)
      .ADD_OPTION(std::vector<kintera::Reaction>,
                  kintera::LindemannFalloffOptionsImpl, reactions)
      .ADD_OPTION(std::vector<double>, kintera::LindemannFalloffOptionsImpl,
                  k0_A)
      .ADD_OPTION(std::vector<double>, kintera::LindemannFalloffOptionsImpl,
                  k0_b)
      .ADD_OPTION(std::vector<double>, kintera::LindemannFalloffOptionsImpl,
                  k0_Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::LindemannFalloffOptionsImpl,
                  kinf_A)
      .ADD_OPTION(std::vector<double>, kintera::LindemannFalloffOptionsImpl,
                  kinf_b)
      .ADD_OPTION(std::vector<double>, kintera::LindemannFalloffOptionsImpl,
                  kinf_Ea_R)
      .ADD_OPTION(std::vector<kintera::Composition>,
                  kintera::LindemannFalloffOptionsImpl, efficiencies);

  ADD_KINTERA_MODULE(LindemannFalloff, LindemannFalloffOptions,
                     &kintera::LindemannFalloffImpl::forward, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"))
      .def("pretty_print", &kintera::LindemannFalloffImpl::pretty_print);

  ////////////// Troe Falloff //////////////
  auto pyTroeFalloffOptions =
      py::class_<kintera::TroeFalloffOptionsImpl, kintera::TroeFalloffOptions>(
          m, "TroeFalloffOptions");

  pyTroeFalloffOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::TroeFalloffOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("TroeFalloffOptions({})", ss.str());
           })
      .ADD_OPTION(double, kintera::TroeFalloffOptionsImpl, Tref)
      .ADD_OPTION(std::string, kintera::TroeFalloffOptionsImpl, units)
      .ADD_OPTION(std::vector<kintera::Reaction>,
                  kintera::TroeFalloffOptionsImpl, reactions)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, k0_A)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, k0_b)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, k0_Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, kinf_A)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, kinf_b)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl,
                  kinf_Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, troe_A)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, troe_T3)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, troe_T1)
      .ADD_OPTION(std::vector<double>, kintera::TroeFalloffOptionsImpl, troe_T2)
      .ADD_OPTION(std::vector<kintera::Composition>,
                  kintera::TroeFalloffOptionsImpl, efficiencies);

  ADD_KINTERA_MODULE(TroeFalloff, TroeFalloffOptions,
                     &kintera::TroeFalloffImpl::forward, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"))
      .def("pretty_print", &kintera::TroeFalloffImpl::pretty_print);

  ////////////// SRI Falloff //////////////
  auto pySRIFalloffOptions =
      py::class_<kintera::SRIFalloffOptionsImpl, kintera::SRIFalloffOptions>(
          m, "SRIFalloffOptions");

  pySRIFalloffOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::SRIFalloffOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("SRIFalloffOptions({})", ss.str());
           })
      .ADD_OPTION(double, kintera::SRIFalloffOptionsImpl, Tref)
      .ADD_OPTION(std::string, kintera::SRIFalloffOptionsImpl, units)
      .ADD_OPTION(std::vector<kintera::Reaction>,
                  kintera::SRIFalloffOptionsImpl, reactions)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, k0_A)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, k0_b)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, k0_Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, kinf_A)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, kinf_b)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl,
                  kinf_Ea_R)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, sri_A)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, sri_B)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, sri_C)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, sri_D)
      .ADD_OPTION(std::vector<double>, kintera::SRIFalloffOptionsImpl, sri_E)
      .ADD_OPTION(std::vector<kintera::Composition>,
                  kintera::SRIFalloffOptionsImpl, efficiencies);

  ADD_KINTERA_MODULE(SRIFalloff, SRIFalloffOptions,
                     &kintera::SRIFalloffImpl::forward, py::arg("temp"),
                     py::arg("pres"), py::arg("conc"), py::arg("other"))
      .def("pretty_print", &kintera::SRIFalloffImpl::pretty_print);

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
      .def_static("from_kinetics_base",
                  &kintera::KineticsOptionsImpl::from_kinetics_base,
                  py::arg("master_input_path"),
                  py::arg("photo_catalog_path") = "", py::arg("cross_dir") = "",
                  py::arg("verbose") = false)
      .def("reactions", &kintera::KineticsOptionsImpl::reactions)
      .ADD_OPTION(double, kintera::KineticsOptionsImpl, Tref)
      .ADD_OPTION(double, kintera::KineticsOptionsImpl, Pref)
      .ADD_OPTION(kintera::ArrheniusOptions, kintera::KineticsOptionsImpl,
                  arrhenius)
      .ADD_OPTION(kintera::CoagulationOptions, kintera::KineticsOptionsImpl,
                  coagulation)
      .ADD_OPTION(kintera::EvaporationOptions, kintera::KineticsOptionsImpl,
                  evaporation)
      .ADD_OPTION(kintera::ThreeBodyOptions, kintera::KineticsOptionsImpl,
                  three_body)
      .ADD_OPTION(kintera::LindemannFalloffOptions,
                  kintera::KineticsOptionsImpl, lindemann_falloff)
      .ADD_OPTION(kintera::TroeFalloffOptions, kintera::KineticsOptionsImpl,
                  troe_falloff)
      .ADD_OPTION(kintera::SRIFalloffOptions, kintera::KineticsOptionsImpl,
                  sri_falloff)
      .ADD_OPTION(bool, kintera::KineticsOptionsImpl, evolve_temperature);

  ADD_KINTERA_MODULE(Kinetics, KineticsOptions,
                     torch::wrap_pybind_function_no_gil(
                         [](kintera::KineticsImpl& self, torch::Tensor temp,
                            torch::Tensor pres, torch::Tensor conc,
                            std::map<std::string, torch::Tensor> const& extra) {
                           return self.forward(temp, pres, conc, extra);
                         }),
                     py::arg("temp"), py::arg("pres"), py::arg("conc"),
                     py::arg("extra") = std::map<std::string, torch::Tensor>{})
      .def(
          "forward_nogil",
          [](kintera::KineticsImpl& self, torch::Tensor temp,
             torch::Tensor pres, torch::Tensor conc,
             std::map<std::string, torch::Tensor> const& extra) {
            py::gil_scoped_release no_gil;
            return self.forward(temp, pres, conc, extra);
          },
          py::arg("temp"), py::arg("pres"), py::arg("conc"),
          py::arg("extra") = std::map<std::string, torch::Tensor>{})
      .def("jacobian", &kintera::KineticsImpl::jacobian, py::arg("temp"),
           py::arg("conc"), py::arg("cvol"), py::arg("rate"), py::arg("rc_ddC"),
           py::arg("rc_ddT") = torch::nullopt);
}
