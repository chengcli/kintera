// kintera
#include "coagulation.hpp"

#include <kintera/kinetics/reaction_species.hpp>
#include <kintera/species.hpp>

namespace kintera {

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        CoagulationOptions op) {
  add_reaction_species(cloud_set, op, false, false);
}

CoagulationOptions CoagulationOptionsImpl::from_yaml(const YAML::Node& root) {
  auto options = CoagulationOptionsImpl::create();
  ArrheniusOptionsImpl::from_yaml(root, options);
  return options;
}

}  // namespace kintera
