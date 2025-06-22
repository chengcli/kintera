// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/log_svp.hpp>

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
    if (node["diff_c"]) {
      options.diff_c().push_back(node["diff_c"].as<double>());
    } else {
      options.diff_c().push_back(0.2);
    }

    if (node["diff_T"]) {
      options.diff_T().push_back(node["diff_T"].as<double>());
    } else {
      options.diff_T().push_back(1.75);
    }

    if (node["diff_P"]) {
      options.diff_P().push_back(node["diff_P"].as<double>());
    } else {
      options.diff_P().push_back(-1.);
    }

    if (node["vm"]) {
      options.vm().push_back(node["vm"].as<double>());
    } else {
      options.vm().push_back(18.);
    }

    if (node["diameter"]) {
      options.diameter().push_back(node["radius"].as<double>());
    } else {
      options.diameter().push_back(1.);
    }

    if (node["minT"]) {
      options.minT().push_back(node["minT"].as<double>());
    } else {
      options.minT().push_back(0.);
    }

    if (node["maxT"]) {
      options.maxT().push_back(node["maxT"].as<double>());
    } else {
      options.maxT().push_back(1.e4);
    }

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
  // log(cm^2/s) -> log(m^2/s)
  log_diff_c = register_buffer(
      "log_diff_c",
      torch::tensor(options.diff_c(), torch::kFloat64).log() + log(1.e-4));

  diff_T = register_buffer("diff_T",
                           torch::tensor(options.diff_T(), torch::kFloat64));
  diff_P = register_buffer("diff_P",
                           torch::tensor(options.diff_P(), torch::kFloat64));

  // log(cm^3/mol) -> log(m^3/mol)
  log_vm = register_buffer(
      "log_vm",
      torch::tensor(options.vm(), torch::kFloat64).log() + log(1.e-6));

  // log(cm) -> log(m)
  log_diameter = register_buffer(
      "log_diameter",
      torch::tensor(options.diameter(), torch::kFloat64).log() + log(1.e-2));
}

void EvaporationImpl::pretty_print(std::ostream& os) const {
  os << "Evaporation Rate: " << std::endl;

  for (size_t i = 0; i < options.diff_c().size(); ++i) {
    os << "(" << i + 1 << ") "
       << "diff_c =" << options.diff_c()[i] << " cm^2/s, "
       << "diff_T =" << options.diff_T()[i] << ", "
       << "diff_P =" << options.diff_P()[i] << ", "
       << "vm =" << options.vm()[i] << " cm^3/mol, "
       << "diameter=" << options.diameter()[i] << " cm" << std::endl;
  }
}

torch::Tensor EvaporationImpl::forward(
    torch::Tensor T, torch::Tensor P,
    std::map<std::string, torch::Tensor> const& other) {
  // expand T if not yet
  torch::Tensor temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;

  auto log_diff = log_diff_c + diff_T * (temp / options.Tref()).log() +
                  diff_P * (P / options.Pref()).log().unsqueeze(-1);

  auto kappa = log(12.) + log_diff + log_vm - 2. * log_diameter;

  std::cout << "kappa = " << kappa << std::endl;

  // saturation deficit
  auto conc = other.at("conc");
  auto stoich = other.at("stoich");
  auto sp = stoich.clamp_min(0.);

  LogSVPFunc::init(options);
  auto logsvp = LogSVPFunc::apply(temp);
  std::cout << "logsvp = " << logsvp << std::endl;

  auto eta = torch::exp(logsvp - sp.sum(0) * (constants::Rgas * temp).log()) -
             conc.unsqueeze(-1).pow(sp).prod(-2);

  return torch::where(eta <= 0, torch::zeros_like(kappa), kappa + eta.log());
}

}  // namespace kintera
