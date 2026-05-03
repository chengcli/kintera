// torch
#include <torch/extension.h>

// pybind11
#include <pybind11/stl.h>

// kintera
#include <kintera/photolysis/actinic_flux.hpp>
#include <kintera/photolysis/photochem.hpp>
#include <kintera/photolysis/photolysis.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_photolysis(py::module& m) {
  ////////////// PhotolysisOptions //////////////
  auto pyPhotolysisOptions =
      py::class_<kintera::PhotolysisOptionsImpl, kintera::PhotolysisOptions>(
          m, "PhotolysisOptions");

  pyPhotolysisOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::PhotolysisOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("PhotolysisOptions({})", ss.str());
           })
      .def_static(
          "from_yaml",
          [](py::object yaml_node) {
            // Convert Python YAML to C++ YAML::Node
            // For now, assume it's already a YAML::Node or handle conversion
            return kintera::PhotolysisOptionsImpl::create();
          },
          py::arg("yaml_node"))
      .ADD_OPTION(std::vector<kintera::Reaction>,
                  kintera::PhotolysisOptionsImpl, reactions)
      .ADD_OPTION(std::vector<double>, kintera::PhotolysisOptionsImpl,
                  wavelength)
      .ADD_OPTION(std::vector<double>, kintera::PhotolysisOptionsImpl,
                  temperature)
      .ADD_OPTION(std::vector<double>, kintera::PhotolysisOptionsImpl,
                  cross_section)
      .ADD_OPTION(std::vector<std::vector<kintera::Composition>>,
                  kintera::PhotolysisOptionsImpl, branches)
      .ADD_OPTION(std::vector<std::vector<std::string>>,
                  kintera::PhotolysisOptionsImpl, branch_names);

  auto pyPhotoChemOptions =
      py::class_<kintera::PhotoChemOptionsImpl, kintera::SpeciesThermoImpl,
                 kintera::PhotoChemOptions>(m, "PhotoChemOptions");

  pyPhotoChemOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::PhotoChemOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("PhotoChemOptions({})", ss.str());
           })
      .def_static("from_yaml",
                  py::overload_cast<std::string const&, bool>(
                      &kintera::PhotoChemOptionsImpl::from_yaml),
                  py::arg("filename"), py::arg("verbose") = false)
      .def_static("from_kinetics_base",
                  &kintera::PhotoChemOptionsImpl::from_kinetics_base,
                  py::arg("master_input_path"),
                  py::arg("photo_catalog_path") = "", py::arg("cross_dir") = "",
                  py::arg("verbose") = false)
      .def("reactions", &kintera::PhotoChemOptionsImpl::reactions)
      .ADD_OPTION(kintera::PhotolysisOptions, kintera::PhotoChemOptionsImpl,
                  photolysis)
      .ADD_OPTION(bool, kintera::PhotoChemOptionsImpl, evolve_temperature);

  ////////////// Photolysis Module //////////////
  torch::python::bind_module<kintera::PhotolysisImpl>(m, "Photolysis")
      .def(py::init<>(), R"(Construct a new default module.)")
      .def(py::init<kintera::PhotolysisOptions>(),
           "Construct a Photolysis module", py::arg("options"))
      .def_readonly("options", &kintera::PhotolysisImpl::options)
      .def("__repr__",
           [](const kintera::PhotolysisImpl& a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("Photolysis(\n{})", ss.str());
           })
      .def("module",
           [](kintera::PhotolysisImpl& self, std::string name) {
             return self.named_modules()[name];
           })
      .def("buffer",
           [](kintera::PhotolysisImpl& self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("forward", &kintera::PhotolysisImpl::forward, py::arg("temp"),
           py::arg("actinic_flux"))
      .def("update_xs_diss_stacked",
           &kintera::PhotolysisImpl::update_xs_diss_stacked, py::arg("temp"))
      .def("interp_cross_section",
           &kintera::PhotolysisImpl::interp_cross_section, py::arg("rxn_idx"),
           py::arg("wave"), py::arg("temp"))
      .def("get_effective_stoich",
           &kintera::PhotolysisImpl::get_effective_stoich, py::arg("rxn_idx"),
           py::arg("wave"), py::arg("aflux"), py::arg("temp"));

  ADD_KINTERA_MODULE(PhotoChem, PhotoChemOptions, py::arg("temp"),
                     py::arg("conc"), py::arg("actinic_flux"))
      .def("jacobian", &kintera::PhotoChemImpl::jacobian, py::arg("conc"),
           py::arg("rate"));

  ////////////// ActinicFluxOptions //////////////
  auto pyActinicFluxOptions =
      py::class_<kintera::ActinicFluxOptionsImpl, kintera::ActinicFluxOptions>(
          m, "ActinicFluxOptions");

  pyActinicFluxOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::ActinicFluxOptions& self) {
             return fmt::format("ActinicFluxOptions(nsrc={})",
                                self->wavelength().size());
           })
      .ADD_OPTION(std::vector<double>, kintera::ActinicFluxOptionsImpl,
                  wavelength)
      .ADD_OPTION(std::vector<double>, kintera::ActinicFluxOptionsImpl,
                  default_flux)
      .ADD_OPTION(double, kintera::ActinicFluxOptionsImpl, wave_min)
      .ADD_OPTION(double, kintera::ActinicFluxOptionsImpl, wave_max);

  ////////////// Helper functions //////////////
  m.def(
      "create_actinic_flux",
      [](kintera::ActinicFluxOptions const& opts, torch::Tensor wavelength) {
        return kintera::create_actinic_flux(opts, wavelength);
      },
      py::arg("options"), py::arg("wavelength"));

  m.def("interpolate_actinic_flux", &kintera::interpolate_actinic_flux,
        py::arg("wavelength"), py::arg("flux"), py::arg("new_wavelength"));

  m.def("create_uniform_flux", &kintera::create_uniform_flux,
        py::arg("wavelength"), py::arg("flux_value"));

  m.def("create_solar_flux", &kintera::create_solar_flux, py::arg("wavelength"),
        py::arg("peak_flux") = 1.e14);
}
