// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/utils/constants.hpp>

#include "arrhenius.hpp"

namespace kintera {

ArrheniusOptions ArrheniusOptions::from_yaml(const YAML::Node& root) {
  ArrheniusOptions options;

  for (auto const& rxn_node : root) {
    if (!rxn_node["type"]) {
      TORCH_CHECK(false, "Reaction type not specified");
    }

    if (rxn_node["type"].as<std::string>() != "Arrhenius") {
      continue;
    }

    auto node = rxn_node["rate-constant"];
    if (node["A"]) {
      options.A().push_back(node["A"].as<double>());
    } else {
      options.A().push_back(1.);
    }

    if (node["b"]) {
      options.b().push_back(node["b"].as<double>());
    } else {
      options.b().push_back(0.);
    }

    if (node["Ea"]) {
      options.Ea_R().push_back(node["Ea"].as<double>());
    } else {
      options.Ea_R().push_back(1.);
    }

    if (node["E4"]) {
      options.E4_R().push_back(node["E4"].as<double>());
    } else {
      options.E4_R().push_back(0.);
    }
  }

  return options;
}

ArrheniusOptions ArrheniusOptions::from_map(
    const std::vector<std::map<std::string, std::string>>& params) {
  ArrheniusOptions options;

  for (auto const& param : params) {
    if (param.count("A")) {
      options.A().push_back(std::stod(param.at("A")));
    } else {
      options.A().push_back(1.);
    }

    if (param.count("b")) {
      options.b().push_back(std::stod(param.at("b")));
    } else {
      options.b().push_back(0.);
    }

    if (param.count("Ea")) {
      options.Ea_R().push_back(std::stod(param.at("Ea")));
    } else {
      options.Ea_R().push_back(1.);
    }

    if (param.count("E4")) {
      options.E4_R().push_back(std::stod(param.at("E4")));
    } else {
      options.E4_R().push_back(0.);
    }
  }

  return options;
}

ArrheniusImpl::ArrheniusImpl(ArrheniusOptions const& options_)
    : options(options_) {
  reset();
}

void ArrheniusImpl::reset() {
  logA = register_buffer("logA",
                         torch::tensor(options.A(), torch::kFloat64).log());
  b = register_buffer("b", torch::tensor(options.b(), torch::kFloat64));
  Ea_R =
      register_buffer("Ea_R", torch::tensor(options.Ea_R(), torch::kFloat64));
  E4_R =
      register_buffer("E4_R", torch::tensor(options.E4_R(), torch::kFloat64));
}

void ArrheniusImpl::pretty_print(std::ostream& os) const {
  os << "Arrhenius Rate: " << std::endl;

  for (size_t i = 0; i < options.A().size(); i++) {
    os << "(" << i + 1 << ") A = " << options.A()[i]
       << ", b = " << options.b()[i]
       << ", Ea = " << options.Ea_R()[i] * constants::GasConstant << " J/mol"
       << std::endl;
  }
}

torch::Tensor ArrheniusImpl::forward(
    torch::Tensor T, std::map<std::string, torch::Tensor> const& other) {
  return logA + b * T.unsqueeze(-1).log() - Ea_R / T.unsqueeze(-1);
}

/*torch::Tensor ArrheniusRate::ddTRate(torch::Tensor T, torch::Tensor P) const {
    return (m_Ea_R * 1.0 / T + m_b) * 1.0 / T;
}*/

}  // namespace kintera
