// torch
#include <torch/extension.h>

// pybind11
#include <pybind11/stl.h>

// kintera
#include <kintera/photolysis/actinic_flux.hpp>
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
           py::arg("wavelength"), py::arg("actinic_flux"))
      .def("interp_cross_section",
           &kintera::PhotolysisImpl::interp_cross_section, py::arg("rxn_idx"),
           py::arg("wave"), py::arg("temp"))
      .def("get_effective_stoich",
           &kintera::PhotolysisImpl::get_effective_stoich, py::arg("rxn_idx"),
           py::arg("wave"), py::arg("aflux"), py::arg("temp"));

  ////////////// ActinicFluxOptions //////////////
  auto pyActinicFluxOptions =
      py::class_<kintera::ActinicFluxOptionsImpl, kintera::ActinicFluxOptions>(
          m, "ActinicFluxOptions");

  pyActinicFluxOptions.def(py::init<>())
      .def("__repr__",
           [](const kintera::ActinicFluxOptions& self) {
             return fmt::format("ActinicFluxOptions(nwave={})",
                                self->wavelength().size());
           })
      .ADD_OPTION(std::vector<double>, kintera::ActinicFluxOptionsImpl,
                  wavelength)
      .ADD_OPTION(std::vector<double>, kintera::ActinicFluxOptionsImpl,
                  default_flux)
      .ADD_OPTION(double, kintera::ActinicFluxOptionsImpl, wave_min)
      .ADD_OPTION(double, kintera::ActinicFluxOptionsImpl, wave_max);

  ////////////// ActinicFluxData //////////////
  py::class_<kintera::ActinicFluxData>(m, "ActinicFluxData")
      .def(py::init<>())
      .def(py::init<torch::Tensor, torch::Tensor>(), py::arg("wavelength"),
           py::arg("flux"))
      .def_readwrite("wavelength", &kintera::ActinicFluxData::wavelength)
      .def_readwrite("flux", &kintera::ActinicFluxData::flux)
      .def("is_valid", &kintera::ActinicFluxData::is_valid)
      .def("nwave", &kintera::ActinicFluxData::nwave)
      .def("interpolate_to", &kintera::ActinicFluxData::interpolate_to,
           py::arg("new_wavelength"))
      .def("to_map", &kintera::ActinicFluxData::to_map)
      .def("__repr__", [](const kintera::ActinicFluxData& self) {
        return fmt::format("ActinicFluxData(nwave={}, valid={})", self.nwave(),
                           self.is_valid());
      });

  ////////////// Helper functions //////////////
  m.def(
      "create_actinic_flux",
      [](kintera::ActinicFluxOptions const& opts, torch::Device device,
         torch::Dtype dtype) {
        return kintera::create_actinic_flux(opts, device, dtype);
      },
      py::arg("options"), py::arg("device") = torch::kCPU,
      py::arg("dtype") = torch::kFloat64);

  m.def(
      "create_uniform_flux",
      [](double wave_min, double wave_max, int nwave, double flux_value) {
        return kintera::create_uniform_flux(wave_min, wave_max, nwave,
                                            flux_value);
      },
      py::arg("wave_min"), py::arg("wave_max"), py::arg("nwave"),
      py::arg("flux_value"));

  m.def(
      "create_uniform_flux",
      [](double wave_min, double wave_max, int nwave, double flux_value,
         torch::Device device, torch::Dtype dtype) {
        return kintera::create_uniform_flux(wave_min, wave_max, nwave,
                                            flux_value, device, dtype);
      },
      py::arg("wave_min"), py::arg("wave_max"), py::arg("nwave"),
      py::arg("flux_value"), py::arg("device") = torch::kCPU,
      py::arg("dtype") = torch::kFloat64);

  m.def(
      "create_solar_flux",
      [](double wave_min, double wave_max, int nwave, double peak_flux) {
        return kintera::create_solar_flux(wave_min, wave_max, nwave, peak_flux);
      },
      py::arg("wave_min"), py::arg("wave_max"), py::arg("nwave"),
      py::arg("peak_flux") = 1.e14);

  m.def(
      "create_solar_flux",
      [](double wave_min, double wave_max, int nwave, double peak_flux,
         torch::Device device, torch::Dtype dtype) {
        return kintera::create_solar_flux(wave_min, wave_max, nwave, peak_flux,
                                          device, dtype);
      },
      py::arg("wave_min"), py::arg("wave_max"), py::arg("nwave"),
      py::arg("peak_flux") = 1.e14, py::arg("device") = torch::kCPU,
      py::arg("dtype") = torch::kFloat64);
}
