// C/C++
#include <cmath>
#include <limits>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/log_svp.hpp>
#include <kintera/units/units.hpp>

#include "evaporation.hpp"
#include "evaporation_extent.hpp"

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<std::string> species_phases;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        EvaporationOptions op) {
  // Resolve phases at parsing time. Existing options subsequently carry their
  // own gas/cloud IDs, independent of the mutable species registry.
  for (auto const& react : op->reactions()) {
    int clouds = 0;
    auto phase_of = [&](std::string const& name) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      auto id = it - species_names.begin();
      return id < species_phases.size() ? species_phases[id] : std::string();
    };
    const bool mixed = react.reactants().size() > 1;
    for (auto const& [name, nu] : react.reactants()) {
      auto phase = phase_of(name);
      TORCH_CHECK(!mixed || !phase.empty(),
                  "Mixed-reactant evaporation requires "
                  "explicit phase declarations: ",
                  name);
      if (phase == "gas") {
        vapor_set.insert(name);
      } else {
        TORCH_CHECK(nu == 1.,
                    "Evaporation requires one condensed reactant "
                    "with coefficient 1: ",
                    react.equation());
        cloud_set.insert(name);
        ++clouds;
      }
    }
    TORCH_CHECK(clouds == 1, "Evaporation requires one condensed reactant: ",
                react.equation());
    for (auto const& [name, nu] : react.products()) {
      auto phase = phase_of(name);
      TORCH_CHECK(phase == "gas" || (!mixed && phase.empty()),
                  "Evaporation products must be gases with explicit phases "
                  "for mixed-reactant reactions: ",
                  name);
      vapor_set.insert(name);
    }
  }
  for (auto const& name : vapor_set)
    TORCH_CHECK(!cloud_set.count(name),
                "Conflicting gas/cloud classification: ", name);
}

EvaporationOptions EvaporationOptionsImpl::from_yaml(const YAML::Node& root) {
  auto options = EvaporationOptionsImpl::create();
  NucleationOptionsImpl::from_yaml(root, options);

  for (const auto& rxn_node : root) {
    if (rxn_node["type"].as<std::string>() != options->name()) continue;

    auto node = rxn_node["rate-constant"];

    auto form = node["form"].as<std::string>("diffusion");
    TORCH_CHECK(form == "diffusion" || form == "kessler",
                "Unknown evaporation form '", form,
                "'; expected 'diffusion' or 'kessler'");
    options->form().push_back(form);
    if (form == "kessler") {
      TORCH_CHECK(node["k3"], "'k3' required for 'kessler' evaporation");
      for (auto key : {"diff_c", "diff_T", "diff_P", "vm", "diameter"})
        TORCH_CHECK(!node[key], "'", key,
                    "' is not used by 'kessler' evaporation");
      double k3 = node["k3"].as<double>();
      TORCH_CHECK(std::isfinite(k3) && k3 >= 0.,
                  "Kessler evaporation 'k3' must be finite and >= 0");
      options->k3().push_back(k3);
    } else {
      TORCH_CHECK(!node["k3"], "'k3' requires form: kessler");
      options->k3().push_back(0.);
    }

    // unit system is [mol, m, s]
    options->diff_c().push_back(node["diff_c"].as<double>(0.2e-4));
    options->diff_T().push_back(node["diff_T"].as<double>(1.75));
    options->diff_P().push_back(node["diff_P"].as<double>(-1.));
    options->vm().push_back(node["vm"].as<double>(18.e-6));
    options->diameter().push_back(node["diameter"].as<double>(1.e-2));
    options->minT().push_back(node["minT"].as<double>(0.));
    options->maxT().push_back(node["maxT"].as<double>(1.e4));
  }

  return options;
}

EvaporationImpl::EvaporationImpl(EvaporationOptions const& options_)
    : options(options_) {
  reset();
}

void EvaporationImpl::reset() {
  diff_c = register_buffer("diff_c",
                           torch::tensor(options->diff_c(), torch::kFloat64));

  diff_T = register_buffer("diff_T",
                           torch::tensor(options->diff_T(), torch::kFloat64));
  diff_P = register_buffer("diff_P",
                           torch::tensor(options->diff_P(), torch::kFloat64));

  vm = register_buffer("vm", torch::tensor(options->vm(), torch::kFloat64));

  diameter = register_buffer(
      "diameter", torch::tensor(options->diameter(), torch::kFloat64));

  std::vector<int64_t> two, general;
  std::set<std::string> condensates;
  auto const& rxns = options->reactions();
  for (size_t j = 0; j < rxns.size(); ++j) {
    auto const& r = rxns[j];
    TORCH_CHECK(
        !r.reversible() && !r.reactants().empty() && !r.products().empty(),
        "Evaporation must be irreversible with reactants and gas products");
    for (auto const& side : {r.reactants(), r.products()})
      for (auto const& [name, nu] : side)
        TORCH_CHECK(std::isfinite(nu) && nu > 0.,
                    "Evaporation coefficients must be finite and positive");
    TORCH_CHECK(r.orders().empty(),
                "Evaporation does not accept custom orders");
    bool simple = r.reactants().size() == 1;
    if (simple) {
      TORCH_CHECK(
          r.reactants().begin()->second == 1.,
          "Evaporation requires one condensed reactant with coefficient 1");
      condensates.insert(r.reactants().begin()->first);
    }
    bool legacy = simple && r.products().size() <= 2;
    for (auto const& [name, nu] : r.products()) {
      legacy &= nu == 1.;
    }
    if (!legacy)
      general.push_back(static_cast<int64_t>(j));
    else if (r.products().size() == 2)
      two.push_back(static_cast<int64_t>(j));
  }
  for (auto const& r : rxns)
    for (auto const& [name, nu] : r.products())
      TORCH_CHECK(!condensates.count(name), "Evaporation reaction '",
                  r.equation(), "' has the condensate '", name,
                  "' as a product");
  two_product_rxns_ = torch::tensor(two, torch::kInt64);
  general_rxns_ = torch::tensor(general, torch::kInt64);

  auto const& form = options->form();
  TORCH_CHECK(
      form.empty() || form.size() == rxns.size(),
      "Evaporation 'form' must be empty or have one entry per reaction");
  TORCH_CHECK(options->k3().empty() || options->k3().size() == rxns.size(),
              "Evaporation 'k3' must be empty or have one entry per reaction");
  std::vector<int64_t> kessler;
  std::vector<double> k3;
  for (size_t j = 0; j < form.size(); ++j) {
    TORCH_CHECK(form[j] == "diffusion" || form[j] == "kessler",
                "Unknown evaporation form '", form[j], "'");
    if (form[j] != "kessler") continue;
    auto const& r = rxns[j];
    TORCH_CHECK(r.reactants().size() == 1 && r.products().size() == 1 &&
                    r.products().begin()->second == 1.,
                "Kessler evaporation requires one condensed reactant and "
                "one gas product, both with coefficient 1: ",
                r.equation());
    TORCH_CHECK(options->k3().size() == rxns.size(),
                "Kessler evaporation requires 'k3'");
    kessler.push_back(static_cast<int64_t>(j));
    k3.push_back(options->k3()[j]);
  }
  kessler_rxns_ = torch::tensor(kessler, torch::kInt64);
  kessler_k3_ = torch::tensor(k3, torch::kFloat64);
}

void EvaporationImpl::pretty_print(std::ostream& os) const {
  os << "Evaporation Rate: " << std::endl;

  for (size_t i = 0; i < options->diff_c().size(); ++i) {
    os << "(" << i + 1 << ") ";
    options->report(os);
  }
}

torch::Tensor EvaporationImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  // expand T if not yet
  auto temp = T;
  if (T.sizes() == P.sizes()) {
    auto vec = T.sizes().vec();
    vec.push_back(diff_c.size(0));
    temp = T.unsqueeze(-1).expand(vec);
  }

  // expand C if not yet
  auto conc = C.dim() == temp.dim() ? C.unsqueeze(-1) : C;

  auto diffusivity = diff_c * (temp / options->Tref()).pow(diff_T) *
                     (P / options->Pref()).unsqueeze(-1).pow(diff_P);

  auto kappa = 12. * diffusivity * vm / (diameter * diameter);

  // saturation deficit
  auto stoich = other.at("stoich");
  auto sp = stoich.clamp_min(0.);
  torch::Tensor gas;
  auto mask = other.find("gas_mask");
  if (mask != other.end()) {
    TORCH_CHECK(mask->second.scalar_type() == torch::kBool &&
                    mask->second.dim() == 1 &&
                    mask->second.size(0) == stoich.size(0),
                "Evaporation gas_mask must be boolean with shape (nspecies,)");
    gas = mask->second.to(stoich.device()).unsqueeze(-1);
  } else {
    for (auto const& r : options->reactions())
      TORCH_CHECK(r.reactants().size() == 1,
                  "Mixed-reactant evaporation requires other[gas_mask]");
    gas = stoich.ge(0.);
  }
  // Kessler form: kappa = k3 / n, n = P/(R T) + the condensed concentrations
  if (kessler_rxns_.numel() > 0) {
    TORCH_CHECK(mask != other.end(),
                "Kessler evaporation requires other[gas_mask]");
    auto condensed = (~gas).to(conc.dtype());
    auto ntot =
        P.unsqueeze(-1) / (constants::Rgas * temp) + (conc * condensed).sum(-2);
    auto idx = kessler_rxns_.to(kappa.device());
    auto k3 = kessler_k3_.to(kappa.device(), kappa.scalar_type());
    kappa = kappa.index_copy(-1, idx, k3 / ntot.index_select(-1, idx));
  }

  auto solid = (~gas) & stoich.ne(0.);
  TORCH_CHECK((solid.sum(0) == 1).all().item<bool>() &&
                  torch::where(solid, stoich.eq(-1), stoich.ge(0) | gas)
                      .all()
                      .item<bool>(),
              "Evaporation requires one condensed reactant with coefficient 1 "
              "and no condensed products");

  LogSVPFunc::init(options);
  auto logsvp = LogSVPFunc::apply(temp);

  auto general_idx = general_rxns_.to(stoich.device());
  auto legacy_logsvp = logsvp;
  if (general_idx.numel() > 0)
    legacy_logsvp = logsvp.index_fill(-1, general_idx, 0.);
  auto ksat =
      torch::exp(legacy_logsvp - sp.sum(0) * (constants::Rgas * temp).log());

  // two-product columns are replaced below; leave their c1 c2 out of eta so an
  // overflowing product cannot send inf * 0 = nan into the gradient
  auto sp1 = sp;
  if (two_product_rxns_.numel() > 0)
    sp1 = sp.index_fill(1, two_product_rxns_.to(sp.device()), 0.);
  if (general_idx.numel() > 0) sp1 = sp1.index_fill(1, general_idx, 0.);
  auto eta = ksat - conc.pow(sp1).prod(-2);

  eta.clamp_min_(0);

  // two products: rate uses the extent x to equilibrium, (c1+x)(c2+x) = ksat
  if (two_product_rxns_.numel() > 0) {
    auto idx = two_product_rxns_.to(ksat.device(), torch::kLong);
    auto k2 = ksat.index_select(-1, idx);
    auto c = conc.clamp_min(0.);
    // an expanded conc carries one column per reaction
    if (c.size(-1) != 1) c = c.index_select(-1, idx);
    auto is_product = sp.index_select(1, idx).gt(0.);
    auto nu = is_product.to(c.dtype());
    // supersaturated columns, including c1 c2 = inf, do not evaporate; zero
    // their concentrations so no inf reaches the extent or its gradient
    auto sub = c.pow(nu).prod(-2).le(k2);
    c = torch::where(sub.unsqueeze(-2), c, torch::zeros_like(c));
    auto prod = c.pow(nu).prod(-2);
    auto sum = torch::where(is_product, c, torch::zeros_like(c)).sum(-2);
    // c1 - c2: +1 on the first product, -1 on the second
    auto sign =
        2. * (is_product & is_product.cumsum(0).eq(1)).to(c.dtype()) - nu;
    auto diff = (c * sign).sum(-2);
    double tiny = c.dtype() == torch::kFloat32
                      ? std::numeric_limits<float>::min()
                      : std::numeric_limits<double>::min();
    // s^2 - 4 c1 c2 = (c1 - c2)^2, without inf - inf
    auto root = (diff * diff + 4. * k2).clamp_min(tiny).sqrt();
    auto x = 2. * (k2 - prod) / (sum + root).clamp_min(tiny);
    x = torch::where(sub, x, torch::zeros_like(x));
    eta = eta.index_copy(-1, idx, x.clamp_min(0.));
  }

  if (general_idx.numel() > 0) {
    auto nu = torch::where(gas, stoich, torch::zeros_like(stoich))
                  .index_select(1, general_idx);
    auto c = conc;
    if (c.size(-1) == 1) {
      auto sizes = c.sizes().vec();
      sizes.back() = general_idx.numel();
      c = c.expand(sizes);
    } else {
      c = c.index_select(-1, general_idx);
    }
    auto tg = temp.index_select(-1, general_idx);
    auto logkc = logsvp.index_select(-1, general_idx) -
                 nu.sum(0) * (constants::Rgas * tg).log();
    auto extent = EvaporationExtent::apply(c, nu, logkc);
    eta = eta.index_copy(-1, general_idx, extent);
  }
  return kappa * eta;
}

}  // namespace kintera
