// yaml-cpp
#include <yaml-cpp/yaml.h>

// kintera
#include "rate_constant.hpp"

namespace kintera {

RateConstantImpl::RateConstantImpl(const RateConstantOptions& options_)
    : options(options_) {
  reset();
}

void RateConstantImpl::reset() {
  YAML::Node root = YAML::LoadFile(options->reaction_file());
  printf("Loading complete\n");

  rxn_id_start.clear();
  rxn_id_end.clear();

  int current_id = 0;
  for (auto const& type : options->types()) {
    int nreaction = 0;
    rxn_id_start.push_back(current_id);

    // TODO: Implement the support of other reaction types
    if (type == "Arrhenius") {
      auto op = ArrheniusOptionsImpl::from_yaml(root);
      eval_rate_constants.push_back(torch::nn::AnyModule(Arrhenius(op)));
      nreaction = op->A().size();
    } else if (type == "three-body") {
      TORCH_CHECK(false, "Three-body reaction not implemented");
    } else if (type == "falloff") {
      TORCH_CHECK(false, "Falloff reaction not implemented");
    } else {
      TORCH_CHECK(false, "Unknown reaction type: ", type);
    }

    current_id += nreaction;
    rxn_id_end.push_back(current_id);
  }

  for (auto& eval_rate_constant : eval_rate_constants) {
    register_module("eval_rate_constant", eval_rate_constant.ptr());
  }
}

torch::Tensor RateConstantImpl::forward(
    torch::Tensor T, std::map<std::string, torch::Tensor> const& other) {
  auto shape = T.sizes().vec();
  for (int i = 0; i < rxn_id_start.size(); i++) {
    std::cout << rxn_id_start[i] << " " << rxn_id_end[i] << std::endl;
  }
  shape.push_back(rxn_id_end.back());

  torch::Tensor result = torch::empty(shape, T.options());

  for (int i = 0; i < eval_rate_constants.size(); i++) {
    result.slice(-1, rxn_id_start[i], rxn_id_end[i]) =
        eval_rate_constants[i].forward(T, other);
  }

  return result;
}

/*torch::Tensor KineticsRatesImpl::forward(torch::Tensor T, torch::Tensor P,
                                         torch::Tensor C) const {
  const auto n_reactions = stoich_matrix_.size(0);
  const auto n_species = stoich_matrix_.size(1);

  auto rate_shapes = C.sizes().vec();
  rate_shapes[0] = n_reactions;
  torch::Tensor rates = torch::zeros(rate_shapes, C.options());

  rates = rates.movedim(0, -1);
  auto result = torch::matmul(rates, stoich_matrix_);
  return result.movedim(-1, 0);
}*/

}  // namespace kintera
