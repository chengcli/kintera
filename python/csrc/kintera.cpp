// pybind11
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// torch
#include <torch/extension.h>

// kintera
#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kintera_formatter.hpp>
#include <kintera/photochem/kinetics_base_reader.hpp>
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

void bind_thermo(py::module& m);
void bind_constants(py::module& m);
void bind_kinetics(py::module& m);
void bind_photolysis(py::module& m);
void bind_diffusion(py::module& m);
void bind_sparse_solver(py::module& m);

PYBIND11_MODULE(kintera, m) {
  m.attr("__name__") = "kintera";
  m.doc() = R"(Atmospheric Thermodynamics and Chemistry Library)";

  auto pySpeciesThermo =
      py::class_<kintera::SpeciesThermoImpl, kintera::SpeciesThermo>(
          m, "SpeciesThermo");

  pySpeciesThermo.def(py::init<>())
      .def("__repr__",
           [](const kintera::SpeciesThermo& self) {
             return fmt::format("SpeciesThermo({})", self);
           })
      .def("species", &kintera::SpeciesThermoImpl::species)
      .def("narrow_copy", &kintera::SpeciesThermoImpl::narrow_copy)
      .def("accumulate", &kintera::SpeciesThermoImpl::accumulate)
      .ADD_OPTION(std::vector<int>, kintera::SpeciesThermoImpl, vapor_ids)
      .ADD_OPTION(std::vector<int>, kintera::SpeciesThermoImpl, cloud_ids)
      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermoImpl, cref_R)
      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermoImpl, uref_R)
      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermoImpl, sref_R);

  auto pyReaction = py::class_<kintera::Reaction>(m, "Reaction");

  pyReaction.def(py::init<>())
      .def(py::init<const std::string&>())
      .def("__repr__",
           [](const kintera::Reaction& self) {
             return fmt::format("Reaction({})", self);
           })
      .def("equation", &kintera::Reaction::equation)
      .ADD_OPTION(kintera::Composition, kintera::Reaction, reactants)
      .ADD_OPTION(kintera::Composition, kintera::Reaction, products);

  auto pyKBAtmosphereProfile =
      py::class_<kintera::KBAtmosphereProfile>(m, "KBAtmosphereProfile");
  pyKBAtmosphereProfile.def(py::init<>())
      .def_readwrite("header", &kintera::KBAtmosphereProfile::header)
      .def_readwrite("altitude", &kintera::KBAtmosphereProfile::altitude)
      .def_readwrite("density", &kintera::KBAtmosphereProfile::density)
      .def_readwrite("temperature",
                     &kintera::KBAtmosphereProfile::temperature)
      .def_readwrite("pressure", &kintera::KBAtmosphereProfile::pressure)
      .def_readwrite("eddy_diffusion",
                     &kintera::KBAtmosphereProfile::eddy_diffusion)
      .def_readwrite("wind", &kintera::KBAtmosphereProfile::wind)
      .def_readwrite("species_profiles",
                     &kintera::KBAtmosphereProfile::species_profiles)
      .def_readwrite(
          "mixing_ratio_species_profiles",
          &kintera::KBAtmosphereProfile::mixing_ratio_species_profiles);

  m.def("parse_kinetics_base_atmosphere",
        &kintera::parse_kinetics_base_atmosphere, py::arg("filepath"));

  auto pyKBPunHeader = py::class_<kintera::KBPunHeader>(m, "KBPunHeader");
  pyKBPunHeader.def(py::init<>())
      .def_readwrite("natom", &kintera::KBPunHeader::natom)
      .def_readwrite("nmol", &kintera::KBPunHeader::nmol)
      .def_readwrite("nreact", &kintera::KBPunHeader::nreact)
      .def_readwrite("npart", &kintera::KBPunHeader::npart)
      .def_readwrite("version", &kintera::KBPunHeader::version);

  auto pyKBPunSpecies =
      py::class_<kintera::KBPunSpecies>(m, "KBPunSpecies");
  pyKBPunSpecies.def(py::init<>())
      .def_readwrite("id", &kintera::KBPunSpecies::id)
      .def_readwrite("name", &kintera::KBPunSpecies::name)
      .def_readwrite("first_reaction",
                     &kintera::KBPunSpecies::first_reaction)
      .def_readwrite("n_reactions", &kintera::KBPunSpecies::n_reactions)
      .def_readwrite("molecular_weight",
                     &kintera::KBPunSpecies::molecular_weight)
      .def_readwrite("composition", &kintera::KBPunSpecies::composition);

  auto pyKBPunParticipant =
      py::class_<kintera::KBPunParticipant>(m, "KBPunParticipant");
  pyKBPunParticipant.def(py::init<>())
      .def_readwrite("coefficient", &kintera::KBPunParticipant::coefficient)
      .def_readwrite("species_id", &kintera::KBPunParticipant::species_id)
      .def_readwrite("marker", &kintera::KBPunParticipant::marker);

  auto pyKBPunRateBlock =
      py::class_<kintera::KBPunRateBlock>(m, "KBPunRateBlock");
  pyKBPunRateBlock.def(py::init<>())
      .def_readwrite("A", &kintera::KBPunRateBlock::A)
      .def_readwrite("b", &kintera::KBPunRateBlock::b)
      .def_readwrite("C", &kintera::KBPunRateBlock::C)
      .def_readwrite("D", &kintera::KBPunRateBlock::D)
      .def_readwrite("E", &kintera::KBPunRateBlock::E)
      .def_readwrite("F", &kintera::KBPunRateBlock::F)
      .def_readwrite("Tmin", &kintera::KBPunRateBlock::Tmin)
      .def_readwrite("Tmax", &kintera::KBPunRateBlock::Tmax)
      .def_readwrite("Fc", &kintera::KBPunRateBlock::Fc)
      .def_readwrite("Tin", &kintera::KBPunRateBlock::Tin)
      .def_readwrite("Tout", &kintera::KBPunRateBlock::Tout);

  auto pyKBPunReaction =
      py::class_<kintera::KBPunReaction>(m, "KBPunReaction");
  pyKBPunReaction.def(py::init<>())
      .def_readwrite("id", &kintera::KBPunReaction::id)
      .def_readwrite("n_reactants", &kintera::KBPunReaction::n_reactants)
      .def_readwrite("n_products", &kintera::KBPunReaction::n_products)
      .def_readwrite("participants", &kintera::KBPunReaction::participants)
      .def_readwrite("rate_blocks", &kintera::KBPunReaction::rate_blocks)
      .def_readwrite("reactant_ids", &kintera::KBPunReaction::reactant_ids)
      .def_readwrite("product_ids", &kintera::KBPunReaction::product_ids)
      .def_readwrite("raw_line", &kintera::KBPunReaction::raw_line);

  auto pyKBPunNetwork =
      py::class_<kintera::KBPunNetwork>(m, "KBPunNetwork");
  pyKBPunNetwork.def(py::init<>())
      .def_readwrite("header", &kintera::KBPunNetwork::header)
      .def_readwrite("elements", &kintera::KBPunNetwork::elements)
      .def_readwrite("species", &kintera::KBPunNetwork::species)
      .def_readwrite("reactions", &kintera::KBPunNetwork::reactions);

  m.def("parse_kinetics_base_pun", &kintera::parse_kinetics_base_pun,
        py::arg("filepath"));

  auto pyKBTitanReactionReport =
      py::class_<kintera::KBTitanReactionReport>(m, "KBTitanReactionReport");
  pyKBTitanReactionReport.def(py::init<>())
      .def_readwrite("total_reactions",
                     &kintera::KBTitanReactionReport::total_reactions)
      .def_readwrite(
          "selected_photolysis_reactions",
          &kintera::KBTitanReactionReport::selected_photolysis_reactions)
      .def_readwrite("thermal_candidate_reactions",
                     &kintera::KBTitanReactionReport::thermal_candidate_reactions)
      .def_readwrite("missing_rate_blocks",
                     &kintera::KBTitanReactionReport::missing_rate_blocks)
      .def_readwrite("charged_species_count",
                     &kintera::KBTitanReactionReport::charged_species_count)
      .def_readwrite("charged_reactions",
                     &kintera::KBTitanReactionReport::charged_reactions)
      .def_readwrite(
          "charged_thermal_candidate_reactions",
          &kintera::KBTitanReactionReport::charged_thermal_candidate_reactions)
      .def_readwrite("ion_mass_action_reactions",
                     &kintera::KBTitanReactionReport::ion_mass_action_reactions)
      .def_readwrite(
          "dissociative_recombination_reactions",
          &kintera::KBTitanReactionReport::dissociative_recombination_reactions)
      .def_readwrite(
          "selected_electron_impact_reactions",
          &kintera::KBTitanReactionReport::selected_electron_impact_reactions)
      .def_readwrite("electron_reactant_reactions",
                     &kintera::KBTitanReactionReport::electron_reactant_reactions)
      .def_readwrite("electron_product_reactions",
                     &kintera::KBTitanReactionReport::electron_product_reactions)
      .def_readwrite("cation_reactant_reactions",
                     &kintera::KBTitanReactionReport::cation_reactant_reactions)
      .def_readwrite("cation_product_reactions",
                     &kintera::KBTitanReactionReport::cation_product_reactions)
      .def_readwrite("anion_reactant_reactions",
                     &kintera::KBTitanReactionReport::anion_reactant_reactions)
      .def_readwrite("anion_product_reactions",
                     &kintera::KBTitanReactionReport::anion_product_reactions)
      .def_readwrite("charge_balanced_reactions",
                     &kintera::KBTitanReactionReport::charge_balanced_reactions)
      .def_readwrite(
          "charge_imbalanced_reactions",
          &kintera::KBTitanReactionReport::charge_imbalanced_reactions)
      .def_readwrite("n_reactants_counts",
                     &kintera::KBTitanReactionReport::n_reactants_counts)
      .def_readwrite("charged_species",
                     &kintera::KBTitanReactionReport::charged_species)
      .def_readwrite("selected_photolysis_ids",
                     &kintera::KBTitanReactionReport::selected_photolysis_ids)
      .def_readwrite(
          "charge_imbalanced_reaction_ids",
          &kintera::KBTitanReactionReport::charge_imbalanced_reaction_ids)
      .def_readwrite("unsupported_reaction_ids",
                     &kintera::KBTitanReactionReport::unsupported_reaction_ids);

  m.def("classify_kinetics_base_titan_reactions",
        py::overload_cast<std::string const&, std::string const&>(
            &kintera::classify_kinetics_base_titan_reactions),
        py::arg("pun_path"), py::arg("run_input_path"));

  bind_thermo(m);
  bind_constants(m);
  bind_kinetics(m);
  bind_photolysis(m);
  bind_diffusion(m);
  bind_sparse_solver(m);

  m.def("species_names", []() -> const std::vector<std::string>& {
    return kintera::species_names;
  });

  m.def("set_species_names", [](const std::vector<std::string>& names) {
    kintera::species_names = names;
    return kintera::species_names;
  });

  m.def("species_weights", []() -> const std::vector<double>& {
    return kintera::species_weights;
  });

  m.def("set_species_weights", [](const std::vector<double>& weights) {
    kintera::species_weights = weights;
    return kintera::species_weights;
  });

  m.def("species_cref_R",
        []() -> const std::vector<double>& { return kintera::species_cref_R; });

  m.def("set_species_cref_R", [](const std::vector<double>& cref_R) {
    kintera::species_cref_R = cref_R;
    return kintera::species_cref_R;
  });

  m.def("species_uref_R",
        []() -> const std::vector<double>& { return kintera::species_uref_R; });

  m.def("set_species_uref_R", [](const std::vector<double>& uref_R) {
    kintera::species_uref_R = uref_R;
    return kintera::species_uref_R;
  });

  m.def("species_sref_R",
        []() -> const std::vector<double>& { return kintera::species_sref_R; });

  m.def("set_species_sref_R", [](const std::vector<double>& sref_R) {
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

  m.def("evolve_ros2", &kintera::evolve_ros2, py::arg("rate1"),
        py::arg("rate2"), py::arg("stoich"), py::arg("jacobian"),
        py::arg("dt"));

  m.def("ros2_k1", &kintera::ros2_k1, py::arg("rate1"), py::arg("stoich"),
        py::arg("jacobian"), py::arg("dt"));
}
