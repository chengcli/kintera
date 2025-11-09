// pybind11
#include <pybind11/stl_bind.h>

// torch
#include <torch/extension.h>

// kintera
#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kintera_formatter.hpp>
#include <kintera/species.hpp>
#include <kintera/utils/find_resource.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
extern std::vector<double> species_cref_R;
extern std::vector<double> species_uref_R;
extern std::vector<double> species_sref_R;

}  // namespace kintera

void bind_thermo(py::module &m);
void bind_constants(py::module &m);
void bind_kinetics(py::module &m);

PYBIND11_MODULE(kintera, m) {
  m.attr("__name__") = "kintera";
  m.doc() = R"(Atmospheric Thermodynamics and Chemistry Library)";

  auto pySpeciesThermo = py::class_<kintera::SpeciesThermo>(m, "SpeciesThermo");

  pySpeciesThermo
      .def(py::init<>())

      .def("__repr__",
           [](const kintera::SpeciesThermo &self) {
             return fmt::format("SpeciesThermo({})", self);
           })

      .def("species", &kintera::SpeciesThermo::species)

      .def("narrow_copy", &kintera::SpeciesThermo::narrow_copy)
      .def("accumulate", &kintera::SpeciesThermo::accumulate)

      .ADD_OPTION(std::vector<int>, kintera::SpeciesThermo, vapor_ids)

      .ADD_OPTION(std::vector<int>, kintera::SpeciesThermo, cloud_ids)

      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermo, cref_R)

      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermo, uref_R)

      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermo, sref_R);

  auto pyReaction = py::class_<kintera::Reaction>(m, "Reaction");

  pyReaction
      .def(py::init<>())

      .def(py::init<const std::string &>())

      .def("__repr__",
           [](const kintera::Reaction &self) {
             return fmt::format("Reaction({})", self);
           })

      .def("equation", &kintera::Reaction::equation)

      .ADD_OPTION(kintera::Composition, kintera::Reaction, reactants)

      .ADD_OPTION(kintera::Composition, kintera::Reaction, products);

  bind_thermo(m);
  bind_constants(m);
  bind_kinetics(m);

  m.def("species_names", []() -> const std::vector<std::string> & {
    return kintera::species_names;
  });

  m.def("set_species_names", [](const std::vector<std::string> &names) {
    kintera::species_names = names;
    return kintera::species_names;
  });

  m.def("species_weights", []() -> const std::vector<double> & {
    return kintera::species_weights;
  });

  m.def("set_species_weights", [](const std::vector<double> &weights) {
    kintera::species_weights = weights;
    return kintera::species_weights;
  });

  m.def("species_cref_R", []() -> const std::vector<double> & {
    return kintera::species_cref_R;
  });

  m.def("set_species_cref_R", [](const std::vector<double> &cref_R) {
    kintera::species_cref_R = cref_R;
    return kintera::species_cref_R;
  });

  m.def("species_uref_R", []() -> const std::vector<double> & {
    return kintera::species_uref_R;
  });

  m.def("set_species_uref_R", [](const std::vector<double> &uref_R) {
    kintera::species_uref_R = uref_R;
    return kintera::species_uref_R;
  });

  m.def("species_sref_R", []() -> const std::vector<double> & {
    return kintera::species_sref_R;
  });

  m.def("set_species_sref_R", [](const std::vector<double> &sref_R) {
    kintera::species_sref_R = sref_R;
    return kintera::species_sref_R;
  });

  m.def(
      "set_search_paths",
      [](const std::string path) {
        strcpy(kintera::search_paths, path.c_str());
        return kintera::deserialize_search_paths(kintera::search_paths);
      },
      py::arg("path"));

  m.def("get_search_paths", []() {
    return kintera::deserialize_search_paths(kintera::search_paths);
  });

  m.def(
      "add_resource_directory",
      [](const std::string path, bool prepend) {
        kintera::add_resource_directory(path, prepend);
        return kintera::deserialize_search_paths(kintera::search_paths);
      },
      py::arg("path"), py::arg("prepend") = true);

  m.def("find_resource", &kintera::find_resource, py::arg("filename"));

  m.def("evolve_implicit", &kintera::evolve_implicit, py::arg("rate"),
        py::arg("stoich"), py::arg("jacobian"), py::arg("dt"));
}
