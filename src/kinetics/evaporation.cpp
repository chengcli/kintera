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
  for (auto& react : op.reactions()) {
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

EvaporationOptions EvaporationOptions::from_yaml(const YAML::Node& root) {
  EvaporationOptions options;

  for (const auto& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != "evaporation") {
      continue;
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");

    std::string equation = rxn_node["equation"].as<std::string>();
    options.reactions().push_back(Reaction(equation));

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");

    auto node = rxn_node["rate-constant"];

    // unit system is [mol, m, s]
    options.diff_c().push_back(node["diff_c"].as<double>(0.2e-4));
    options.diff_T().push_back(node["diff_T"].as<double>(1.75));
    options.diff_P().push_back(node["diff_P"].as<double>(-1.));
    options.vm().push_back(node["vm"].as<double>(18.e-6));
    options.diameter().push_back(node["diameter"].as<double>(1.e-2));
    options.minT().push_back(node["minT"].as<double>(0.));
    options.maxT().push_back(node["maxT"].as<double>(1.e4));

    TORCH_CHECK(node["formula"],
                "'formula' is not defined in the rate-constant");

    auto formula = node["formula"].as<std::string>();

    TORCH_CHECK(get_user_func1().find(formula) != get_user_func1().end(),
                "Formula '", formula, "' is not defined in the user functions");
    options.logsvp().push_back(get_user_func1()[formula]);

    TORCH_CHECK(
        get_user_func1().find(formula + "_ddT") != get_user_func1().end(),
        "Formula '", formula, "' is not defined in the user functions");
    options.logsvp_ddT().push_back(get_user_func1()[formula + "_ddT"]);
  }

  return options;
}

EvaporationImpl::EvaporationImpl(EvaporationOptions const& options_)
    : options(std::move(options_)) {
  reset();
}

void EvaporationImpl::reset() {
  diff_c = register_buffer("diff_c",
                           torch::tensor(options.diff_c(), torch::kFloat64));

  diff_T = register_buffer("diff_T",
                           torch::tensor(options.diff_T(), torch::kFloat64));
  diff_P = register_buffer("diff_P",
                           torch::tensor(options.diff_P(), torch::kFloat64));

  vm = register_buffer("vm", torch::tensor(options.vm(), torch::kFloat64));

  diameter = register_buffer(
      "diameter", torch::tensor(options.diameter(), torch::kFloat64));
}

void EvaporationImpl::pretty_print(std::ostream& os) const {
  os << "Evaporation Rate: " << std::endl;

  for (size_t i = 0; i < options.diff_c().size(); ++i) {
    os << "(" << i + 1 << ") ";
    options.report(os);
  }
}

torch::Tensor EvaporationImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  // expand T if not yet
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;

  // expand C if not yet
  auto conc = C.dim() == temp.dim() ? C.unsqueeze(-1) : C;

  auto diffusivity = diff_c * (temp / options.Tref()).pow(diff_T) *
                     (P / options.Pref()).unsqueeze(-1).pow(diff_P);

  auto kappa = 12. * diffusivity * vm / (diameter * diameter);

  // saturation deficit
  auto stoich = other.at("stoich");
  auto sp = stoich.clamp_min(0.);

  LogSVPFunc::init(options);
  auto logsvp = LogSVPFunc::apply(temp);

  auto eta = torch::exp(logsvp - sp.sum(0) * (constants::Rgas * temp).log()) -
             conc.pow(sp).prod(-2);

  eta.clamp_min_(0);

  return kappa * eta;
}

}  // namespace kintera
