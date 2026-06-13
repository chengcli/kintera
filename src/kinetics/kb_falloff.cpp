// C/C++
#include <algorithm>
#include <cmath>

// fmt
#include <fmt/format.h>

#include "kb_falloff.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, KBFalloffOptions op) {
  for (auto& react : op->reactions()) {
    for (auto& [name, _] : react.reactants()) {
      if (name == "M" || name == "(+M)") continue;
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name, " not found");
      vapor_set.insert(name);
    }
    for (auto& [name, _] : react.products()) {
      if (name == "M" || name == "(+M)") continue;
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name, " not found");
      vapor_set.insert(name);
    }
  }
}

KBFalloffImpl::KBFalloffImpl(KBFalloffOptions const& options_)
    : options(options_) {
  reset();
}

void KBFalloffImpl::reset() {
  int nreaction = options->reactions().size();
  if (nreaction == 0) return;

  k0_A =
      register_buffer("k0_A", torch::tensor(options->k0_A(), torch::kFloat64));
  k0_b =
      register_buffer("k0_b", torch::tensor(options->k0_b(), torch::kFloat64));
  k0_Ea_R = register_buffer("k0_Ea_R",
                            torch::tensor(options->k0_Ea_R(), torch::kFloat64));
  kinf_A = register_buffer("kinf_A",
                           torch::tensor(options->kinf_A(), torch::kFloat64));
  kinf_b = register_buffer("kinf_b",
                           torch::tensor(options->kinf_b(), torch::kFloat64));
  kinf_Ea_R = register_buffer(
      "kinf_Ea_R", torch::tensor(options->kinf_Ea_R(), torch::kFloat64));
}

void KBFalloffImpl::pretty_print(std::ostream& os) const {
  os << "KB Falloff Rate (fc = " << options->fc() << "):" << std::endl;
  for (size_t i = 0; i < options->reactions().size(); i++) {
    os << "(" << i + 1 << ") " << options->reactions()[i].equation()
       << std::endl;
    os << "    k_low:  A = " << options->k0_A()[i]
       << ", b = " << options->k0_b()[i] << ", Ea_R = " << options->k0_Ea_R()[i]
       << " K" << std::endl;
    os << "    k_high: A = " << options->kinf_A()[i]
       << ", b = " << options->kinf_b()[i]
       << ", Ea_R = " << options->kinf_Ea_R()[i] << " K" << std::endl;
  }
}

torch::Tensor KBFalloffImpl::compute_arrhenius(torch::Tensor T, torch::Tensor A,
                                               torch::Tensor b,
                                               torch::Tensor Ea_R) const {
  auto Tref = options->Tref();
  return A * (T / Tref).pow(b) * torch::exp(-Ea_R / T);
}

torch::Tensor KBFalloffImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  int nreaction = options->reactions().size();
  if (nreaction == 0) {
    auto out_shape = T.sizes().vec();
    out_shape.push_back(0);
    return torch::empty(out_shape, T.options());
  }

  // expand T to broadcast against the reaction dim if not yet
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;

  auto k_low = compute_arrhenius(temp, k0_A, k0_b, k0_Ea_R);
  auto k_high = compute_arrhenius(temp, kinf_A, kinf_b, kinf_Ea_R);

  // Total number density n. Prefer an explicit field; else sum over species.
  torch::Tensor n;
  auto it = other.find("number_density");
  if (it != other.end()) {
    n = it->second;
  } else {
    torch::Tensor C_actual;
    if (C.dim() >= 2 && C.size(-1) == nreaction) {
      // C has shape (..., nspecies, nreaction); all reaction copies identical
      C_actual = C.select(C.dim() - 1, 0);  // (..., nspecies)
    } else {
      C_actual = C;  // (..., nspecies)
    }
    n = C_actual.sum(-1);  // (...)
  }
  // align n with the reaction dim
  auto n_e = n.unsqueeze(-1);

  // KB falloff blend with fc broadening
  auto ratio = k_low * n_e / k_high;
  double fc = options->fc();
  auto logr = torch::log10(ratio.clamp_min(1e-300));
  auto F = torch::exp((std::log(fc)) / (1.0 + logr * logr));
  auto result = (k_low / (1.0 + ratio)) * F;

  // Match the validated reference: rate is zero where k_low or k_high <= 0.
  auto positive = (k_low > 0.0).logical_and(k_high > 0.0);
  return torch::where(positive, result, torch::zeros_like(result));
}

}  // namespace kintera
