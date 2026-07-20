// C/C++
#include <algorithm>

// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/units/units.hpp>

#include "arrhenius.hpp"

namespace kintera {

extern std::vector<std::string> species_names;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ArrheniusOptions op) {
  for (auto& react : op->reactions()) {
    // go through reactants
    for (auto& [name, _] : react.reactants()) {
      auto it = std::find(species_names.begin(), species_names.end(), name);
      TORCH_CHECK(it != species_names.end(), "Species ", name,
                  " not found in species list");
      vapor_set.insert(name);
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

ArrheniusOptions ArrheniusOptionsImpl::from_yaml(
    const YAML::Node& root,
    std::shared_ptr<ArrheniusOptionsImpl> derived_type_ptr) {
  auto options =
      derived_type_ptr ? derived_type_ptr : ArrheniusOptionsImpl::create();

  // Optional multi-range (temperature-banded) buffers. Collected per reaction
  // and only committed to the options if at least one reaction supplies ranges,
  // so legacy single-range configs stay on the exact single-range path.
  std::vector<std::vector<double>> Ar, br, er, tr;
  bool any_ranged = false;

  for (auto const& rxn_node : root) {
    TORCH_CHECK(rxn_node["type"], "Reaction type not specified");

    if (rxn_node["type"].as<std::string>() != options->name()) {
      continue;  // skip this reaction
    }

    TORCH_CHECK(rxn_node["equation"],
                "'equation' is not defined in the reaction");

    std::string equation = rxn_node["equation"].as<std::string>();
    options->reactions().push_back(Reaction(equation));

    // calcualte sum of reactant stoichiometric coefficients
    double sum_stoich = 0.;
    for (const auto& [_, coeff] : options->reactions().back().reactants()) {
      sum_stoich += coeff;
    }

    TORCH_CHECK(rxn_node["rate-constant"],
                "'rate-constant' is not defined in the reaction");

    auto node = rxn_node["rate-constant"];

    // default unit system is [mol, m, s]
    UnitSystem us;

    // input unit system is [molecule, cm, s]
    // [A] []^a []^b ... = molecule cm^-3 s^-1
    // [A] = molecule^(1 - a - b - ...) cm^(-3(1 - a - b - ...)) s^-1
    auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - sum_stoich,
                            -3. * (1. - sum_stoich));
    if (node["A"]) {
      options->A().push_back(us.convert_from(node["A"].as<double>(), unit));
    } else {
      options->A().push_back(1.);
    }

    options->b().push_back(node["b"].as<double>(0.));
    options->Ea_R().push_back(node["Ea_R"].as<double>(1.));
    options->E4_R().push_back(node["E4"].as<double>(0.));

    // Optional multi-range parameters. A reaction may replace the single (A,b,
    // Ea_R) with temperature bands via `A_ranges` + `T_ranges` (and optional
    // `b_ranges`, `Ea_R_ranges`). `T_ranges` lists the ascending upper bound of
    // each band; band r covers [T_ranges[r-1], T_ranges[r]), the last -> +inf.
    // The rate within a band is the usual A*(T/Tref)^b*exp(-Ea_R/T).
    if (node["A_ranges"]) {
      any_ranged = true;
      auto av = node["A_ranges"].as<std::vector<double>>();
      for (auto& a : av) a = us.convert_from(a, unit);
      TORCH_CHECK(node["T_ranges"],
                  "reaction with 'A_ranges' must also define 'T_ranges'");
      auto tv = node["T_ranges"].as<std::vector<double>>();
      TORCH_CHECK(tv.size() == av.size(),
                  "'A_ranges' and 'T_ranges' must have equal length");
      std::vector<double> zeros(av.size(), 0.);
      auto bv =
          node["b_ranges"] ? node["b_ranges"].as<std::vector<double>>() : zeros;
      auto ev = node["Ea_R_ranges"]
                    ? node["Ea_R_ranges"].as<std::vector<double>>()
                    : zeros;
      TORCH_CHECK(bv.size() == av.size() && ev.size() == av.size(),
                  "'b_ranges'/'Ea_R_ranges' must match 'A_ranges' length");
      Ar.push_back(av);
      tr.push_back(tv);
      br.push_back(bv);
      er.push_back(ev);
    } else {
      // single-range placeholder (kept aligned 1:1 with reactions in case some
      // other reaction in this block is ranged); a 1-band range covering all T.
      Ar.push_back({options->A().back()});
      tr.push_back({1.0e30});
      br.push_back({options->b().back()});
      er.push_back({options->Ea_R().back()});
    }
  }

  // Only activate the multi-range path if at least one reaction supplied
  // ranges; otherwise leave A_ranges empty so reset() takes the bit-identical
  // single path.
  if (any_ranged) {
    options->A_ranges() = Ar;
    options->b_ranges() = br;
    options->Ea_R_ranges() = er;
    options->T_ranges() = tr;
  }

  return options;
}

ArrheniusImpl::ArrheniusImpl(ArrheniusOptions const& options_)
    : options(options_) {
  reset();
}

void ArrheniusImpl::reset() {
  // legacy single-range buffers (kept for reporting / introspection)
  A = register_buffer("A", torch::tensor(options->A(), torch::kFloat64));
  b = register_buffer("b", torch::tensor(options->b(), torch::kFloat64));
  Ea_R =
      register_buffer("Ea_R", torch::tensor(options->Ea_R(), torch::kFloat64));
  E4_R =
      register_buffer("E4_R", torch::tensor(options->E4_R(), torch::kFloat64));

  // Sentinels bounding the first/last range. Physical temperatures are
  // positive and well below 1e30 K, so range 0 always covers low T and the
  // top range always covers high T.
  constexpr double T_LO_SENTINEL = 0.0;
  constexpr double T_HI_SENTINEL = 1.0e30;

  const bool multi = !options->A_ranges().empty();
  const int nreaction =
      multi ? (int)options->A_ranges().size() : (int)options->A().size();

  // Maximum number of temperature ranges across all reactions (>= 1).
  nrange = 1;
  if (multi) {
    for (auto const& a : options->A_ranges()) {
      nrange = std::max<int>(nrange, (int)a.size());
    }
  }

  // Padded (nreaction, nrange) parameter tables. Padded slots get A = 0 and a
  // never-matching [T_HI, T_HI) window so they contribute nothing.
  std::vector<double> a_flat(nreaction * nrange, 0.0);
  std::vector<double> b_flat(nreaction * nrange, 0.0);
  std::vector<double> e_flat(nreaction * nrange, 0.0);
  std::vector<double> tlo_flat(nreaction * nrange, T_HI_SENTINEL);
  std::vector<double> thi_flat(nreaction * nrange, T_HI_SENTINEL);

  for (int i = 0; i < nreaction; ++i) {
    if (multi) {
      auto const& av = options->A_ranges()[i];
      auto const& bv = options->b_ranges()[i];
      auto const& ev = options->Ea_R_ranges()[i];
      auto const& tv = options->T_ranges()[i];
      const int ni = (int)av.size();
      TORCH_CHECK(ni >= 1, "Arrhenius reaction ", i,
                  " has no temperature range");
      TORCH_CHECK((int)bv.size() == ni && (int)ev.size() == ni,
                  "Arrhenius multi-range A/b/Ea_R length mismatch at reaction ",
                  i);
      TORCH_CHECK(
          (int)tv.size() == ni,
          "Arrhenius T_ranges must give one upper bound per range at reaction ",
          i);
      double prev = T_LO_SENTINEL;
      for (int r = 0; r < ni; ++r) {
        a_flat[i * nrange + r] = av[r];
        b_flat[i * nrange + r] = bv[r];
        e_flat[i * nrange + r] = ev[r];
        tlo_flat[i * nrange + r] = prev;
        thi_flat[i * nrange + r] = (r == ni - 1) ? T_HI_SENTINEL : tv[r];
        prev = tv[r];
      }
    } else {
      a_flat[i * nrange] = options->A()[i];
      b_flat[i * nrange] = options->b()[i];
      e_flat[i * nrange] = options->Ea_R()[i];
      tlo_flat[i * nrange] = T_LO_SENTINEL;
      thi_flat[i * nrange] = T_HI_SENTINEL;
    }
  }

  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  Amr = register_buffer("Amr",
                        torch::tensor(a_flat, opt).view({nreaction, nrange}));
  bmr = register_buffer("bmr",
                        torch::tensor(b_flat, opt).view({nreaction, nrange}));
  Ea_Rmr = register_buffer(
      "Ea_Rmr", torch::tensor(e_flat, opt).view({nreaction, nrange}));
  Tlo = register_buffer("Tlo",
                        torch::tensor(tlo_flat, opt).view({nreaction, nrange}));
  Thi = register_buffer("Thi",
                        torch::tensor(thi_flat, opt).view({nreaction, nrange}));
}

void ArrheniusImpl::pretty_print(std::ostream& os) const {
  os << "Arrhenius Rate: " << std::endl;

  for (size_t i = 0; i < options->A().size(); i++) {
    os << "(" << i + 1 << ") A = " << options->A()[i]
       << ", b = " << options->b()[i] << ", Ea_R = " << options->Ea_R()[i]
       << " K" << std::endl;
  }
}

torch::Tensor ArrheniusImpl::forward(
    torch::Tensor T, torch::Tensor P, torch::Tensor C,
    std::map<std::string, torch::Tensor> const& other) {
  // expand T to be broadcastable against the reaction dim if not yet
  auto temp = T.sizes() == P.sizes() ? T.unsqueeze(-1) : T;

  // add a trailing range dim: (..., 1, 1) or (..., nreaction, 1) so it
  // broadcasts against the (nreaction, nrange) parameter tables.
  auto tempr = temp.unsqueeze(-1);

  // candidate rate per (reaction, range): (..., nreaction, nrange)
  auto rate_rr =
      Amr * (tempr / options->Tref()).pow(bmr) * torch::exp(-Ea_Rmr / tempr);

  // single-range fast path: bit-identical to the legacy single-range result
  if (nrange == 1) {
    return rate_rr.squeeze(-1);
  }

  // select the range whose [Tlo, Thi) window contains the local temperature
  auto mask = (tempr >= Tlo).logical_and(tempr < Thi).to(rate_rr.dtype());
  return (rate_rr * mask).sum(-1);
}

}  // namespace kintera
