// C/C++
#include <limits>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/log_svp.hpp>
#include <kintera/units/units.hpp>

#include "evaporation.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        EvaporationOptions op) {
  for (auto& react : op->reactions()) {
    // go through reactants
    for (auto& [name, _] : react.reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      cloud_set.insert(name);
    }

    // go through products
    for (auto& [name, _] : react.products()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      vapor_set.insert(name);
    }
  }
}

EvaporationOptions EvaporationOptionsImpl::from_yaml(const YAML::Node& root) {
  auto options = EvaporationOptionsImpl::create();
  NucleationOptionsImpl::from_yaml(root, options);

  for (const auto& rxn_node : root) {
    if (rxn_node["type"].as<std::string>() != options->name()) continue;

    auto node = rxn_node["rate-constant"];

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

  // the rate law needs one reactant and one or two products, all coefficient 1
  std::vector<int64_t> two;
  std::set<std::string> condensates;
  auto const& rxns = options->reactions();
  for (size_t j = 0; j < rxns.size(); ++j) {
    auto const& r = rxns[j];
    bool unit_reactant = r.reactants().size() == 1 &&
                         r.reactants().begin()->second == 1. && !r.reversible();
    bool unit_products = r.products().size() == 1 || r.products().size() == 2;
    for (auto const& [name, nu] : r.products()) unit_products &= (nu == 1.);
    TORCH_CHECK(unit_reactant && unit_products, "Evaporation reaction '",
                r.equation(),
                "' must be irreversible with one reactant and one or two "
                "products, each with stoichiometric coefficient 1");
    condensates.insert(r.reactants().begin()->first);
    if (r.products().size() == 2) two.push_back(static_cast<int64_t>(j));
  }
  for (auto j : two)
    for (auto const& [name, _] : rxns[j].products())
      TORCH_CHECK(!condensates.count(name), "Evaporation reaction '",
                  rxns[j].equation(), "' has the condensate '", name,
                  "' as a product");
  two_product_rxns_ = torch::tensor(two, torch::kInt64);
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

  LogSVPFunc::init(options);
  auto logsvp = LogSVPFunc::apply(temp);

  auto ksat = torch::exp(logsvp - sp.sum(0) * (constants::Rgas * temp).log());
  auto eta = ksat - conc.pow(sp).prod(-2);

  eta.clamp_min_(0);

  // two products: rate uses the extent x to equilibrium, (c1+x)(c2+x) = ksat
  if (two_product_rxns_.numel() > 0) {
    auto idx = two_product_rxns_.to(ksat.device(), torch::kLong);
    auto k2 = ksat.index_select(-1, idx);
    auto c = conc.clamp_min(0.);
    // an expanded conc carries one column per reaction
    if (c.size(-1) != 1) c = c.index_select(-1, idx);
    auto is_product = sp.index_select(1, idx).gt(0.);
    auto prod = c.pow(is_product.to(c.dtype())).prod(-2);
    auto sum = torch::where(is_product, c, torch::zeros_like(c)).sum(-2);
    double tiny = c.dtype() == torch::kFloat32
                      ? std::numeric_limits<float>::min()
                      : std::numeric_limits<double>::min();
    auto root = ((sum * sum - 4. * prod).clamp_min(0.) + 4. * k2)
                    .clamp_min(tiny)
                    .sqrt();
    auto x = 2. * (k2 - prod) / (sum + root).clamp_min(tiny);
    eta = eta.index_copy(-1, idx, x.clamp_min(0.));
  }

  return kappa * eta;
}

}  // namespace kintera
