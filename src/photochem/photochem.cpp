// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/photochem/kinetics_base_reader.hpp>

#include "photochem.hpp"

namespace kintera {

extern std::vector<std::string> species_names;
extern bool species_initialized;

extern std::vector<double> species_cref_R;
extern std::vector<double> species_uref_R;
extern std::vector<double> species_sref_R;
extern std::vector<std::array<double, 9>> species_nasa9_low;
extern std::vector<std::array<double, 9>> species_nasa9_high;
extern std::vector<double> species_nasa9_Tmid;

std::shared_ptr<PhotoChemImpl> PhotoChemImpl::create(
    PhotoChemOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p, "[PhotoChem] Parent module is null");
  TORCH_CHECK(opts, "[PhotoChem] Options pointer is null");
  return p->register_module(name, PhotoChem(opts));
}

PhotoChemOptions PhotoChemOptionsImpl::from_yaml(std::string const& filename,
                                                 bool verbose) {
  auto config = YAML::LoadFile(filename);
  if (!config["reference-state"]) return nullptr;

  if (!species_initialized) {
    init_species_from_yaml(filename);
  }

  return PhotoChemOptionsImpl::from_yaml(config, verbose);
}

PhotoChemOptions PhotoChemOptionsImpl::from_yaml(YAML::Node const& config,
                                                 bool verbose) {
  if (!config["reference-state"]) return nullptr;
  if (!species_initialized) {
    init_species_from_yaml(config);
  }

  auto photo = PhotoChemOptionsImpl::create();
  photo->verbose(verbose);

  if (config["reactions"]) {
    photo->photolysis() = PhotolysisOptionsImpl::from_yaml(config["reactions"]);
  }

  for (int id = 0; id < (int)species_names.size(); ++id) {
    photo->vapor_ids().push_back(id);
  }

  for (const auto& id : photo->vapor_ids()) {
    photo->cref_R().push_back(species_cref_R[id]);
    photo->uref_R().push_back(species_uref_R[id]);
    photo->sref_R().push_back(species_sref_R[id]);
    photo->nasa9_low().push_back(species_nasa9_low[id]);
    photo->nasa9_high().push_back(species_nasa9_high[id]);
    photo->nasa9_Tmid().push_back(species_nasa9_Tmid[id]);
  }

  return photo;
}

PhotoChemOptions PhotoChemOptionsImpl::from_kinetics_base(
    std::string const& master_input_path, std::string const& photo_catalog_path,
    std::string const& cross_dir, bool verbose) {
  return photochem_options_from_kinetics_base(
      master_input_path, photo_catalog_path, cross_dir, verbose);
}

std::vector<Reaction> const& PhotoChemOptionsImpl::reactions() const {
  static const std::vector<Reaction> empty_reactions;
  if (!photolysis()) return empty_reactions;
  return photolysis()->reactions();
}

PhotoChemImpl::PhotoChemImpl(const PhotoChemOptions& options_)
    : options(options_) {
  populate_thermo(options);
  reset();
}

void PhotoChemImpl::reset() {
  auto species = options->species();
  auto const& reactions = options->reactions();
  auto nspecies = species.size();
  auto nreaction = reactions.size();

  auto stoich_data =
      torch::zeros({(int)nspecies, (int)nreaction}, torch::kFloat64);
  auto react_data =
      torch::zeros({(int)nspecies, (int)nreaction}, torch::kFloat64);

  for (int j = 0; j < (int)nreaction; ++j) {
    auto const& r = reactions[j];
    for (int i = 0; i < (int)species.size(); ++i) {
      auto react_it = r.reactants().find(species[i]);
      if (react_it != r.reactants().end()) {
        stoich_data[i][j] -= react_it->second;
        react_data[i][j] = react_it->second;
      }
      auto prod_it = r.products().find(species[i]);
      if (prod_it != r.products().end()) {
        stoich_data[i][j] += prod_it->second;
      }
    }
  }

  stoich = register_buffer("stoich", stoich_data);
  react_stoich_ = register_buffer("react_stoich", react_data);

  photolysis_evaluator = Photolysis(options->photolysis());
  register_module("photolysis", photolysis_evaluator);
}

torch::Tensor PhotoChemImpl::forward(torch::Tensor temp, torch::Tensor conc,
                                     torch::Tensor actinic_flux) {
  auto nrxn = options->reactions().size();
  auto out_shape = temp.sizes().vec();
  out_shape.push_back(nrxn);
  if (nrxn == 0) {
    return torch::empty(out_shape, temp.options());
  }

  photolysis_evaluator->update_xs_diss_stacked(temp);
  auto k = photolysis_evaluator->forward(temp, actinic_flux);
  auto conc_safe = conc.clamp_min(1e-300);
  return k * conc_safe.unsqueeze(-1).pow(react_stoich_).prod(-2);
}

torch::Tensor PhotoChemImpl::jacobian(torch::Tensor conc,
                                      torch::Tensor rate) const {
  auto conc_safe = conc.clamp_min(1e-300);
  auto react_st = react_stoich_.t();
  return react_st * rate.unsqueeze(-1) / conc_safe.unsqueeze(-2);
}

}  // namespace kintera
