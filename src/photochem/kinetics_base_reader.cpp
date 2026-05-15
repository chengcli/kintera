// C/C++
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

// fmt
#include <fmt/format.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/photochem/photochem.hpp>
#include <kintera/units/units.hpp>

#include "kinetics_base_reader.hpp"

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
extern std::vector<double> species_cref_R;
extern std::vector<double> species_uref_R;
extern std::vector<double> species_sref_R;
extern bool species_initialized;

extern std::vector<std::array<double, 9>> species_nasa9_low;
extern std::vector<std::array<double, 9>> species_nasa9_high;
extern std::vector<double> species_nasa9_Tmid;

static std::string trim(std::string const& s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

static std::string to_upper(std::string s) {
  for (auto& c : s) c = std::toupper(c);
  return s;
}

static std::string normalize_species_name(std::string const& raw) {
  std::string name = trim(raw);
  // ^1CH2 -> (1)CH2, ^13C4H6 -> (13)C4H6
  std::regex caret_re(R"(^\^(\d+)(.*))");
  std::smatch m;
  if (std::regex_match(name, m, caret_re)) {
    name = "(" + m[1].str() + ")" + m[2].str();
  }
  // C2H7^+ -> C2H7+, C2H7^- -> C2H7-
  size_t pos;
  while ((pos = name.find("^+")) != std::string::npos)
    name.replace(pos, 2, "+");
  while ((pos = name.find("^-")) != std::string::npos)
    name.replace(pos, 2, "-");
  return name;
}

static std::vector<double> extract_sci_numbers(std::string const& s) {
  std::vector<double> result;
  std::regex num_re(R"([-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+\.\d*)");
  std::sregex_iterator it(s.begin(), s.end(), num_re);
  std::sregex_iterator end;
  for (; it != end; ++it) {
    try {
      result.push_back(std::stod(it->str()));
    } catch (...) {
    }
  }
  return result;
}

static bool has_composition(std::string const& line,
                            std::set<std::string> const& elements) {
  for (auto const& elem : elements) {
    std::string pattern = "\\b" + elem + "\\s*=\\s*\\d+";
    std::regex re(pattern);
    std::smatch m;
    if (std::regex_search(line, m, re)) {
      std::string before = line.substr(0, m.position());
      auto comment_pos = before.find('!');
      if (comment_pos != std::string::npos)
        before = before.substr(0, comment_pos);
      // reaction lines have space-plus-space separator before the match
      std::regex plus_re(R"(\s\+\s)");
      if (!std::regex_search(before, plus_re)) {
        return true;
      }
    }
  }
  return false;
}

static KBSpecies parse_species_line(
    std::string const& line, std::map<std::string, double> const& elem_masses) {
  KBSpecies sp;

  auto comment_pos = line.find('!');
  std::string data_part =
      (comment_pos != std::string::npos) ? line.substr(0, comment_pos) : line;

  // Extract species name: first non-whitespace token
  std::istringstream iss(data_part);
  std::string first_token;
  iss >> first_token;
  if (first_token.empty()) return sp;
  sp.name = normalize_species_name(first_token);

  // Parse elemental composition
  for (auto const& [elem, mass] : elem_masses) {
    std::string pattern = "\\b" + elem + "\\s*=\\s*(\\d+)";
    std::regex re(pattern);
    std::smatch m;
    if (std::regex_search(data_part, m, re)) {
      int count = std::stoi(m[1].str());
      if (count > 0) sp.composition[elem] = count;
    }
  }

  // Compute molecular weight
  sp.molecular_weight = 0.0;
  for (auto const& [elem, count] : sp.composition) {
    auto it = elem_masses.find(elem);
    if (it != elem_masses.end()) {
      sp.molecular_weight += it->second * count;
    }
  }

  // Parse heat of formation
  std::regex hf_re(R"(HF\s*=\s*([-+]?\d+\.?\d*))");
  std::smatch hf_m;
  if (std::regex_search(data_part, hf_m, hf_re)) {
    sp.hf_kcal = std::stod(hf_m[1].str());
  }

  // Parse NASA-9 thermodynamic data
  std::regex nt_re(R"(NT\s*=\s*(\d+))");
  std::smatch nt_m;
  if (std::regex_search(data_part, nt_m, nt_re)) {
    sp.n_nasa9_ranges = std::stoi(nt_m[1].str());

    std::regex tr_re(R"(TR\s*=\s*([\d.Ee+-]+))");
    std::regex ta_re(R"(TA\s*=\s*((?:\s*[-+]?\d+\.?\d*[eE][-+]?\d+){7}))");
    std::regex tb_re(R"(TB\s*=\s*((?:\s*[-+]?\d+\.?\d*[eE][-+]?\d+){2}))");

    std::vector<double> tr_vals;
    for (std::sregex_iterator it(data_part.begin(), data_part.end(), tr_re);
         it != std::sregex_iterator(); ++it) {
      tr_vals.push_back(std::stod((*it)[1].str()));
    }

    std::vector<std::vector<double>> ta_vals;
    for (std::sregex_iterator it(data_part.begin(), data_part.end(), ta_re);
         it != std::sregex_iterator(); ++it) {
      ta_vals.push_back(extract_sci_numbers((*it)[1].str()));
    }

    std::vector<std::vector<double>> tb_vals;
    for (std::sregex_iterator it(data_part.begin(), data_part.end(), tb_re);
         it != std::sregex_iterator(); ++it) {
      tb_vals.push_back(extract_sci_numbers((*it)[1].str()));
    }

    // Store first range as "low", second as "high"
    if (ta_vals.size() >= 1 && ta_vals[0].size() == 7 && tb_vals.size() >= 1 &&
        tb_vals[0].size() == 2) {
      for (int k = 0; k < 7; ++k) sp.nasa9_low[k] = ta_vals[0][k];
      sp.nasa9_low[7] = tb_vals[0][0];
      sp.nasa9_low[8] = tb_vals[0][1];
    }

    if (ta_vals.size() >= 2 && ta_vals[1].size() == 7 && tb_vals.size() >= 2 &&
        tb_vals[1].size() == 2) {
      for (int k = 0; k < 7; ++k) sp.nasa9_high[k] = ta_vals[1][k];
      sp.nasa9_high[7] = tb_vals[1][0];
      sp.nasa9_high[8] = tb_vals[1][1];
    }

    if (tr_vals.size() >= 2) {
      sp.nasa9_Tmid = tr_vals[1];
    }
  }

  return sp;
}

struct ParsedReactionLine {
  std::string equation;
  std::vector<double> rate_nums;
  std::string comment;
};

static ParsedReactionLine extract_equation_and_rates(std::string const& line) {
  ParsedReactionLine result;

  auto comment_pos = line.find('!');
  std::string data_part =
      (comment_pos != std::string::npos) ? line.substr(0, comment_pos) : line;
  if (comment_pos != std::string::npos)
    result.comment = trim(line.substr(comment_pos + 1));

  // Truncate at '>' (multi-temperature-range separator)
  auto gt_pos = data_part.find('>');
  if (gt_pos != std::string::npos) data_part = data_part.substr(0, gt_pos);

  auto eq_pos = data_part.find('=');
  if (eq_pos == std::string::npos) return result;

  std::string after_eq = data_part.substr(eq_pos + 1);

  // Find start of rate constants: first scientific notation number not
  // embedded in a species name
  std::regex sci_re(R"([-+]?\d+\.?\d*[eE][-+]?\d+)");
  std::sregex_iterator it(after_eq.begin(), after_eq.end(), sci_re);
  std::sregex_iterator end;

  size_t rate_start = std::string::npos;
  for (; it != end; ++it) {
    auto pos = static_cast<size_t>(it->position());
    if (pos > 0) {
      char prev = after_eq[pos - 1];
      if (prev != ' ' && prev != '\t' && prev != '+' && prev != ',') continue;
    }
    rate_start = eq_pos + 1 + pos;
    break;
  }

  if (rate_start != std::string::npos) {
    result.equation = trim(data_part.substr(0, rate_start));
    std::string rate_str = data_part.substr(rate_start);
    result.rate_nums = extract_sci_numbers(rate_str);
    // Also capture plain decimal numbers in the rate section
    std::regex all_num_re(R"([-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+\.\d*|\d+)");
    std::vector<double> all_nums;
    std::sregex_iterator rit(rate_str.begin(), rate_str.end(), all_num_re);
    for (; rit != std::sregex_iterator(); ++rit) {
      try {
        all_nums.push_back(std::stod(rit->str()));
      } catch (...) {
      }
    }
    result.rate_nums = all_nums;
  } else {
    result.equation = trim(data_part);
  }

  return result;
}

static std::pair<std::vector<std::string>, std::vector<std::string>>
parse_equation_string(std::string const& eq_str) {
  std::vector<std::string> reactants, products;

  auto eq_pos = eq_str.find('=');
  if (eq_pos == std::string::npos) return {reactants, products};

  std::string lhs = eq_str.substr(0, eq_pos);
  std::string rhs = eq_str.substr(eq_pos + 1);

  auto parse_side = [](std::string const& s) {
    std::vector<std::string> species;
    std::istringstream iss(s);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token) tokens.push_back(token);

    for (size_t i = 0; i < tokens.size(); ++i) {
      if (tokens[i] == "+") continue;

      std::string sp = tokens[i];
      // Check for stoichiometric prefix: "2O" or "2O2"
      std::regex stoich_re(R"(^(\d+)([A-Za-z(^].*)$)");
      std::smatch m;
      if (std::regex_match(sp, m, stoich_re)) {
        int count = std::stoi(m[1].str());
        std::string name = normalize_species_name(m[2].str());
        for (int j = 0; j < count; ++j) species.push_back(name);
      } else {
        species.push_back(normalize_species_name(sp));
      }
    }
    return species;
  };

  reactants = parse_side(lhs);
  products = parse_side(rhs);
  return {reactants, products};
}

static std::vector<int> extract_ints(std::string const& s) {
  std::vector<int> result;
  std::regex int_re(R"([-+]?\d+)");
  for (std::sregex_iterator it(s.begin(), s.end(), int_re);
       it != std::sregex_iterator(); ++it) {
    try {
      result.push_back(std::stoi(it->str()));
    } catch (...) {
    }
  }
  return result;
}

static double parse_fortran_double(std::string text) {
  if (text.find('e') == std::string::npos &&
      text.find('E') == std::string::npos) {
    for (size_t i = 1; i < text.size(); ++i) {
      if ((text[i] == '+' || text[i] == '-') &&
          std::isdigit(static_cast<unsigned char>(text[i - 1]))) {
        text.insert(i, "E");
        break;
      }
    }
  }
  return std::stod(text);
}

static std::vector<double> extract_doubles(std::string const& s) {
  std::vector<double> result;
  std::regex number_re(
      R"([-+]?(?:\d+\.?\d*|\.\d+)(?:(?:[eE][-+]?\d+)|(?:[-+]\d{2,3}))?)");
  for (std::sregex_iterator it(s.begin(), s.end(), number_re);
       it != std::sregex_iterator(); ++it) {
    try {
      result.push_back(parse_fortran_double(it->str()));
    } catch (...) {
    }
  }
  return result;
}

static bool parse_pun_species_header(std::string const& line,
                                     KBPunSpecies& species) {
  static std::regex const header_prefix_re(R"(^\s*\d+\.\s+)");
  if (!std::regex_search(line, header_prefix_re)) return false;

  std::istringstream iss(line);
  int id = 0;
  char dot = '\0';
  std::string name;
  int first_reaction = 0;
  int n_reactions = 0;
  double molecular_weight = 0.0;

  if (!(iss >> id >> dot) || dot != '.') return false;
  if (!(iss >> name >> first_reaction >> n_reactions >> molecular_weight)) {
    return false;
  }

  species = {};
  species.id = id;
  species.name = normalize_species_name(name);
  species.first_reaction = first_reaction;
  species.n_reactions = n_reactions;
  species.molecular_weight = molecular_weight;

  int count = 0;
  while (iss >> count) {
    species.composition.push_back(count);
  }
  return true;
}

static bool parse_pun_reaction_header(std::string const& line,
                                      KBPunReaction& reaction) {
  static std::regex const header_prefix_re(R"(^\s*\d+\.\s+)");
  if (!std::regex_search(line, header_prefix_re)) return false;

  std::istringstream iss(line);
  int id = 0;
  char dot = '\0';
  int n_reactants = 0;
  int n_products = 0;

  if (!(iss >> id >> dot) || dot != '.') return false;
  if (!(iss >> n_reactants >> n_products)) return false;

  reaction = {};
  reaction.id = id;
  reaction.n_reactants = n_reactants;
  reaction.n_products = n_products;
  reaction.raw_line = line;

  std::regex sci_re(R"([-+]?\d+\.?\d*[eE][-+]?\d+)");
  auto rate_match = std::sregex_iterator(line.begin(), line.end(), sci_re);
  size_t participant_end =
      rate_match == std::sregex_iterator()
          ? line.size()
          : static_cast<size_t>(rate_match->position());
  auto participant_text = line.substr(0, participant_end);
  std::regex participant_re(R"(\(\s*(\d*)\s*\)\s*(\d+)\s*([+=]?))");
  bool on_products = false;
  for (std::sregex_iterator it(participant_text.begin(), participant_text.end(),
                               participant_re);
       it != std::sregex_iterator(); ++it) {
    KBPunParticipant participant;
    participant.coefficient =
        (*it)[1].str().empty() ? 1 : std::stoi((*it)[1].str());
    if (participant.coefficient <= 0) participant.coefficient = 1;
    participant.species_id = std::stoi((*it)[2].str());
    participant.marker =
        (*it)[3].str().empty() ? ' ' : static_cast<char>((*it)[3].str()[0]);
    reaction.participants.push_back(participant);

    if (participant.species_id != 0) {
      auto& ids = on_products ? reaction.product_ids : reaction.reactant_ids;
      for (int i = 0; i < participant.coefficient; ++i) {
        ids.push_back(participant.species_id);
      }
    }
    if (participant.marker == '=') on_products = true;
  }

  auto values = extract_doubles(line.substr(participant_end));
  auto make_rate_block = [](std::vector<double> const& values, size_t offset) {
    KBPunRateBlock block;
    auto get = [&](size_t index, double fallback) {
      return offset + index < values.size() ? values[offset + index] : fallback;
    };
    block.A = get(0, 0.0);
    block.b = get(1, 0.0);
    block.C = get(2, 0.0);
    block.D = get(3, 0.0);
    block.E = get(4, 0.0);
    block.F = get(5, 0.0);
    block.Tmin = get(6, 0.0);
    block.Tmax = get(7, 0.0);
    block.Fc = get(8, 1.0);
    block.Tin = get(9, 0.0);
    block.Tout = get(10, 0.0);
    return block;
  };
  for (size_t offset = 0; offset < values.size(); offset += 11) {
    reaction.rate_blocks.push_back(make_rate_block(values, offset));
  }

  if (reaction.reactant_ids.empty() && reaction.product_ids.empty()) {
    auto eq_pos = participant_text.find('=');
    if (eq_pos == std::string::npos) return true;
    auto expand_ids = [](std::string const& side) {
    std::vector<int> ids;
    std::regex participant_re(R"(\(\s*(\d*)\s*\)\s*(\d+))");
    for (std::sregex_iterator it(side.begin(), side.end(), participant_re);
         it != std::sregex_iterator(); ++it) {
      int coeff = (*it)[1].str().empty() ? 1 : std::stoi((*it)[1].str());
      int species_id = std::stoi((*it)[2].str());
      if (species_id == 0) continue;
      for (int i = 0; i < coeff; ++i) ids.push_back(species_id);
    }
    return ids;
    };
    reaction.reactant_ids = expand_ids(participant_text.substr(0, eq_pos));
    reaction.product_ids = expand_ids(participant_text.substr(eq_pos + 1));
  }
  return true;
}

static std::vector<int> collect_int_block(std::vector<std::string> const& lines,
                                          int start_idx, int count) {
  std::vector<int> values;
  for (int i = start_idx; i < (int)lines.size() && (int)values.size() < count;
       ++i) {
    std::string stripped = trim(lines[i]);
    if (stripped.empty()) continue;
    if (stripped.find(':') != std::string::npos && !values.empty()) break;

    auto ints = extract_ints(stripped);
    for (int value : ints) {
      values.push_back(value);
      if ((int)values.size() == count) break;
    }
  }
  return values;
}

KBPunNetwork parse_kinetics_base_pun(std::string const& filepath) {
  KBPunNetwork network;

  std::ifstream ifs(filepath);
  TORCH_CHECK(ifs.good(), "Cannot open KINETICS-base .pun file: ", filepath);

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) lines.push_back(line);

  TORCH_CHECK(lines.size() >= 3, "Invalid KINETICS-base .pun file: ",
              filepath);

  {
    std::istringstream header_iss(lines[1]);
    header_iss >> network.header.natom >> network.header.nmol >>
        network.header.nreact >> network.header.npart >>
        network.header.version;
  }

  TORCH_CHECK(network.header.natom > 0 && network.header.nmol > 0 &&
                  network.header.nreact > 0,
              "Invalid KINETICS-base .pun header in: ", filepath);

  int line_idx = 2;
  while (line_idx < (int)lines.size() &&
         (int)network.elements.size() < network.header.natom) {
    std::istringstream elem_iss(lines[line_idx]);
    std::string symbol;
    double mass = 0.0;
    while (elem_iss >> symbol >> mass) {
      network.elements[to_upper(symbol)] = mass;
      if ((int)network.elements.size() == network.header.natom) break;
    }
    ++line_idx;
  }

  TORCH_CHECK((int)network.elements.size() == network.header.natom,
              "Could not parse all KINETICS-base .pun elements in: ",
              filepath);

  for (; line_idx < (int)lines.size() &&
         (int)network.species.size() < network.header.nmol;
       ++line_idx) {
    KBPunSpecies species;
    if (parse_pun_species_header(lines[line_idx], species)) {
      network.species.push_back(std::move(species));
    }
  }

  TORCH_CHECK((int)network.species.size() == network.header.nmol,
              "Could not parse all KINETICS-base .pun species in: ",
              filepath);

  for (; line_idx < (int)lines.size() &&
         (int)network.reactions.size() < network.header.nreact;
       ++line_idx) {
    KBPunReaction reaction;
    if (parse_pun_reaction_header(lines[line_idx], reaction)) {
      network.reactions.push_back(std::move(reaction));
    }
  }

  TORCH_CHECK((int)network.reactions.size() == network.header.nreact,
              "Could not parse all KINETICS-base .pun reactions in: ",
              filepath);

  return network;
}

KBRunSelection parse_kinetics_base_run_input(std::string const& filepath) {
  KBRunSelection selection;

  std::ifstream ifs(filepath);
  TORCH_CHECK(ifs.good(), "Cannot open KINETICS-base run input: ", filepath);

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) lines.push_back(line);

  for (int i = 0; i + 1 < (int)lines.size(); ++i) {
    std::string label = to_upper(trim(lines[i]));
    if (label.find("NFIX NVARYS NVARYF") != std::string::npos) {
      auto values = extract_ints(lines[i + 1]);
      TORCH_CHECK(values.size() >= 3,
                  "Could not parse NFIX/NVARYS/NVARYF from: ", filepath);
      selection.nfix = values[0];
      selection.nvarys = values[1];
      selection.nvaryf = values[2];
    } else if (label.find("NPHOTO NPHOTS NPHOTR NPHOTD") !=
               std::string::npos) {
      auto values = extract_ints(lines[i + 1]);
      TORCH_CHECK(values.size() >= 4,
                  "Could not parse NPHOTO/NPHOTS/NPHOTR/NPHOTD from: ",
                  filepath);
      selection.nphoto = values[0];
      selection.nphots = values[1];
      selection.nphotr = values[2];
      selection.nphotd = values[3];
    } else if (label.find("IFIX,IVARYS,IVARYF") != std::string::npos) {
      int total = selection.nfix + selection.nvarys + selection.nvaryf;
      auto values = collect_int_block(lines, i + 1, total);
      TORCH_CHECK((int)values.size() == total,
                  "Could not parse all selected species ids from: ", filepath);

      selection.fixed_species_ids.assign(values.begin(),
                                         values.begin() + selection.nfix);
      selection.varying_slow_species_ids.assign(
          values.begin() + selection.nfix,
          values.begin() + selection.nfix + selection.nvarys);
      selection.varying_fast_species_ids.assign(
          values.begin() + selection.nfix + selection.nvarys, values.end());
    } else if (label.find("IPHOTO,IPHOTS,IPHOTR,IPHOTD") !=
               std::string::npos) {
      int total =
          selection.nphoto + selection.nphots + selection.nphotr +
          selection.nphotd;
      selection.photolysis_reaction_ids = collect_int_block(lines, i + 1, total);
      TORCH_CHECK((int)selection.photolysis_reaction_ids.size() == total,
                  "Could not parse all selected photolysis ids from: ",
                  filepath);
    }
  }

  TORCH_CHECK(selection.nfix + selection.nvarys + selection.nvaryf > 0,
              "No KINETICS-base species selection found in: ", filepath);

  return selection;
}

KBAtmosphereProfile parse_kinetics_base_atmosphere(
    std::string const& filepath) {
  KBAtmosphereProfile profile;

  std::ifstream ifs(filepath);
  TORCH_CHECK(ifs.good(), "Cannot open KINETICS-base atmosphere file: ",
              filepath);

  std::string current_section;
  std::map<std::string, std::vector<double>> sections;
  std::string line;
  while (std::getline(ifs, line)) {
    auto stripped = trim(line);
    if (stripped.empty()) continue;

    if (stripped[0] == '%') {
      current_section = trim(stripped.substr(1));
      bool is_mixing_ratio = false;
      if (!current_section.empty() && current_section.back() == '&') {
        current_section.pop_back();
        is_mixing_ratio = true;
      }
      current_section = normalize_species_name(trim(current_section));
      sections[current_section] = {};
      if (is_mixing_ratio) {
        profile.mixing_ratio_species_profiles.push_back(current_section);
      }
      continue;
    }

    if (current_section.empty()) {
      if (profile.header.empty()) profile.header = stripped;
      continue;
    }

    auto values = extract_doubles(stripped);
    auto& section_values = sections[current_section];
    section_values.insert(section_values.end(), values.begin(), values.end());
  }

  auto take_required = [&](std::string const& name) {
    auto it = sections.find(name);
    TORCH_CHECK(it != sections.end(), "Missing KINETICS-base atmosphere section %",
                name, " in: ", filepath);
    auto values = std::move(it->second);
    sections.erase(it);
    return values;
  };

  profile.altitude = take_required("ALT");
  auto nlevel = profile.altitude.size();
  TORCH_CHECK(nlevel > 0, "Empty KINETICS-base atmosphere profile in: ",
              filepath);

  auto take_center_values = [&](std::string const& name) {
    auto values = take_required(name);
    TORCH_CHECK(values.size() >= nlevel, "KINETICS-base atmosphere section %",
                name, " has ", values.size(), " values, expected at least ",
                nlevel, " in: ", filepath);
    values.resize(nlevel);
    return values;
  };

  profile.density = take_center_values("DEN");
  profile.temperature = take_center_values("TEMP");
  profile.pressure = take_center_values("PRE");
  profile.eddy_diffusion = take_center_values("EDDY");
  profile.wind = take_center_values("WIND");
  profile.species_profiles = std::move(sections);

  for (auto const& [species, values] : profile.species_profiles) {
    TORCH_CHECK(values.size() == nlevel, "KINETICS-base species profile %",
                species, " has ", values.size(), " values, expected ", nlevel,
                " in: ", filepath);
  }

  return profile;
}

KBMasterData parse_kinetics_base_master(std::string const& filepath) {
  KBMasterData data;

  std::ifstream ifs(filepath);
  TORCH_CHECK(ifs.good(), "Cannot open master input: ", filepath);

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) lines.push_back(line);

  // Find STOP marker
  int stop_idx = -1;
  for (int i = 0; i < (int)lines.size(); ++i) {
    if (trim(to_upper(lines[i])) == "STOP") {
      stop_idx = i;
      break;
    }
  }
  TORCH_CHECK(stop_idx >= 0, "No STOP marker found in master input");

  // Parse element block
  for (int i = 0; i < stop_idx; ++i) {
    std::istringstream iss(lines[i]);
    std::string sym;
    double mass;
    while (iss >> sym >> mass) {
      data.elements[to_upper(sym)] = mass;
    }
  }

  std::set<std::string> elem_set;
  for (auto const& [k, _] : data.elements) elem_set.insert(k);

  // Parse species and reactions after STOP
  for (int i = stop_idx + 1; i < (int)lines.size(); ++i) {
    std::string stripped = trim(lines[i]);
    if (stripped.empty() || stripped[0] == '!') continue;

    if (has_composition(lines[i], elem_set)) {
      auto sp = parse_species_line(lines[i], data.elements);
      if (!sp.name.empty()) {
        data.species.push_back(sp);
      }
      continue;
    }

    auto parsed = extract_equation_and_rates(lines[i]);
    if (parsed.equation.empty() ||
        parsed.equation.find('=') == std::string::npos)
      continue;

    auto [reactants, products] = parse_equation_string(parsed.equation);
    if (reactants.empty()) continue;

    // Skip pseudo-reactions like "X = PROD"
    if (products.size() == 1 && products[0] == "PROD") continue;

    bool has_M =
        std::find(reactants.begin(), reactants.end(), "M") != reactants.end();

    bool is_photo =
        (parsed.rate_nums.size() >= 3 &&
         std::all_of(parsed.rate_nums.begin(), parsed.rate_nums.begin() + 3,
                     [](double x) { return std::abs(x) < 1e-30; }) &&
         !has_M) ||
        parsed.rate_nums.size() < 3;

    if (is_photo) {
      KBReaction rxn;
      rxn.reactants = reactants;
      rxn.products = products;
      data.photolysis.push_back(rxn);
    } else if (has_M) {
      KBReaction rxn;
      rxn.has_M = true;
      // Remove M from reactants and products
      rxn.reactants.reserve(reactants.size());
      for (auto const& r : reactants)
        if (r != "M") rxn.reactants.push_back(r);
      rxn.products.reserve(products.size());
      for (auto const& p : products)
        if (p != "M") rxn.products.push_back(p);

      if (parsed.rate_nums.size() >= 6) {
        rxn.has_kinf = true;
        rxn.A0 = parsed.rate_nums[0];
        rxn.b0 = parsed.rate_nums[1];
        rxn.Ea_R0 = parsed.rate_nums[2];
        rxn.A_inf = parsed.rate_nums[3];
        rxn.b_inf = parsed.rate_nums[4];
        rxn.Ea_R_inf = parsed.rate_nums[5];
      } else if (parsed.rate_nums.size() >= 3) {
        rxn.has_kinf = false;
        rxn.A0 = parsed.rate_nums[0];
        rxn.b0 = parsed.rate_nums[1];
        rxn.Ea_R0 = parsed.rate_nums[2];
      } else {
        continue;
      }
      data.thermal.push_back(rxn);
    } else {
      if (parsed.rate_nums.size() < 3) continue;
      KBReaction rxn;
      rxn.reactants = reactants;
      rxn.products = products;
      rxn.A = parsed.rate_nums[0];
      rxn.b = parsed.rate_nums[1];
      rxn.Ea_R = parsed.rate_nums[2];
      data.thermal.push_back(rxn);
    }
  }

  return data;
}

std::vector<std::pair<std::string, std::string>> parse_kinetics_base_catalog(
    std::string const& filepath) {
  std::vector<std::pair<std::string, std::string>> entries;

  std::ifstream ifs(filepath);
  if (!ifs.good()) return entries;

  std::string line;
  while (std::getline(ifs, line)) {
    if (trim(line).empty()) continue;
    // Catalog format: equation in columns 0-59, filename in column 60+
    std::string eq_part, fname_part;
    if (line.size() > 60) {
      eq_part = trim(line.substr(0, 60));
      fname_part = trim(line.substr(60));
    } else {
      // Try splitting by multiple spaces
      auto last_space = line.find_last_of(" \t");
      if (last_space != std::string::npos) {
        eq_part = trim(line.substr(0, last_space));
        fname_part = trim(line.substr(last_space));
      }
    }
    if (!eq_part.empty() && !fname_part.empty()) {
      entries.emplace_back(eq_part, fname_part);
    }
  }

  return entries;
}

KBCrossSectionFile parse_kinetics_base_cross_section(
    std::string const& filepath) {
  KBCrossSectionFile result;

  std::ifstream ifs(filepath);
  if (!ifs.good()) return result;

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) lines.push_back(line);

  if (lines.size() < 5) return result;

  result.equation = trim(lines[0]);

  int n_datasets = 0;
  try {
    n_datasets = std::stoi(trim(lines[1]));
  } catch (...) {
    return result;
  }

  int line_idx = 2;
  for (int d = 0; d < n_datasets && line_idx + 1 < (int)lines.size(); ++d) {
    KBCrossSection ds;

    // Parse metadata line 1: type temperature
    {
      std::istringstream iss(lines[line_idx]);
      double type_f;
      if (!(iss >> type_f)) break;
      ds.type = static_cast<int>(type_f);
      if (!(iss >> ds.temperature)) ds.temperature = 298.0;
      ++line_idx;
    }

    // Parse metadata line 2: start_bin end_bin (or just n_points)
    int n_points = 0;
    {
      if (line_idx >= (int)lines.size()) break;
      std::istringstream iss(lines[line_idx]);
      std::vector<int> meta_ints;
      int v;
      while (iss >> v) meta_ints.push_back(v);
      if (meta_ints.size() >= 2 && meta_ints[0] > 0 &&
          meta_ints[1] > meta_ints[0]) {
        n_points = meta_ints[1] - meta_ints[0] + 1;
      } else if (!meta_ints.empty()) {
        n_points = meta_ints.back();
      }
      ++line_idx;
    }

    // Read data lines
    for (int p = 0; p < n_points && line_idx < (int)lines.size(); ++p) {
      std::istringstream iss(lines[line_idx]);
      double wl, val;
      if (iss >> wl >> val) {
        ds.wavelengths_nm.push_back(wl * 0.1);  // Angstrom -> nm
        ds.values.push_back(val);
      } else {
        break;
      }
      ++line_idx;
    }

    if (!ds.wavelengths_nm.empty()) {
      result.datasets.push_back(std::move(ds));
    }
  }

  return result;
}

KBTitanNetwork parse_kinetics_base_titan(
    std::string const& pun_path, std::string const& run_input_path,
    std::string const& photo_catalog_path, std::string const& cross_dir) {
  KBTitanNetwork network;
  network.pun = parse_kinetics_base_pun(pun_path);
  network.selection = parse_kinetics_base_run_input(run_input_path);
  network.catalog = parse_kinetics_base_catalog(photo_catalog_path);

  std::set<int> reaction_ids;
  for (auto const& reaction : network.pun.reactions) {
    reaction_ids.insert(reaction.id);
  }
  for (int id : network.selection.photolysis_reaction_ids) {
    if (reaction_ids.count(id) == 0) {
      network.missing_selected_photolysis_ids.push_back(id);
    }
  }

  std::string cross_prefix = cross_dir;
  if (!cross_prefix.empty() && cross_prefix.back() != '/') {
    cross_prefix += "/";
  }
  for (auto const& [_, filename] : network.catalog) {
    auto path = cross_prefix + filename;
    auto csf = parse_kinetics_base_cross_section(path);
    if (csf.datasets.empty()) {
      network.missing_cross_section_files.push_back(path);
    } else {
      ++network.resolved_cross_sections;
    }
  }

  return network;
}

static int kinetics_base_species_charge(std::string const& name) {
  if (name == "E") return -1;
  if (!name.empty() && name.back() == '+') return 1;
  if (!name.empty() && name.back() == '-') return -1;
  return 0;
}

static std::vector<std::string> species_names_from_ids(
    std::vector<int> const& ids,
    std::map<int, std::string> const& species_by_id) {
  std::vector<std::string> names;
  for (int id : ids) {
    auto it = species_by_id.find(id);
    if (it != species_by_id.end()) {
      names.push_back(it->second);
    }
  }
  return names;
}

static int total_charge(std::vector<std::string> const& names) {
  int charge = 0;
  for (auto const& name : names) {
    charge += kinetics_base_species_charge(name);
  }
  return charge;
}

static bool has_electron(std::vector<std::string> const& names) {
  return std::find(names.begin(), names.end(), "E") != names.end();
}

static bool has_positive_ion(std::vector<std::string> const& names) {
  return std::any_of(names.begin(), names.end(), [](std::string const& name) {
    return kinetics_base_species_charge(name) > 0;
  });
}

static bool has_negative_ion(std::vector<std::string> const& names) {
  return std::any_of(names.begin(), names.end(), [](std::string const& name) {
    return name != "E" && kinetics_base_species_charge(name) < 0;
  });
}

KBTitanReactionReport classify_kinetics_base_titan_reactions(
    KBTitanNetwork const& titan) {
  KBTitanReactionReport report;
  report.total_reactions = static_cast<int>(titan.pun.reactions.size());

  std::map<int, std::string> species_by_id;
  for (auto const& species : titan.pun.species) {
    species_by_id[species.id] = species.name;
    if (kinetics_base_species_charge(species.name) != 0) {
      report.charged_species.push_back(species.name);
    }
  }
  report.charged_species_count =
      static_cast<int>(report.charged_species.size());

  std::set<int> selected_photo_ids(
      titan.selection.photolysis_reaction_ids.begin(),
      titan.selection.photolysis_reaction_ids.end());

  for (auto const& reaction : titan.pun.reactions) {
    report.n_reactants_counts[reaction.n_reactants]++;
    if (reaction.rate_blocks.empty()) {
      report.missing_rate_blocks++;
    }

    auto reactants = species_names_from_ids(reaction.reactant_ids, species_by_id);
    auto products = species_names_from_ids(reaction.product_ids, species_by_id);
    int reactant_charge = total_charge(reactants);
    int product_charge = total_charge(products);
    bool charged = reactant_charge != 0 || product_charge != 0 ||
                   has_electron(reactants) || has_electron(products) ||
                   has_positive_ion(reactants) || has_positive_ion(products) ||
                   has_negative_ion(reactants) || has_negative_ion(products);
    if (charged) {
      report.charged_reactions++;
      if (reactant_charge == product_charge) {
        report.charge_balanced_reactions++;
      } else {
        report.charge_imbalanced_reactions++;
        report.charge_imbalanced_reaction_ids.push_back(reaction.id);
      }
    }
    if (has_electron(reactants)) report.electron_reactant_reactions++;
    if (has_electron(products)) report.electron_product_reactions++;
    if (has_positive_ion(reactants)) report.cation_reactant_reactions++;
    if (has_positive_ion(products)) report.cation_product_reactions++;
    if (has_negative_ion(reactants)) report.anion_reactant_reactions++;
    if (has_negative_ion(products)) report.anion_product_reactions++;

    if (selected_photo_ids.count(reaction.id) != 0) {
      report.selected_photolysis_reactions++;
      report.selected_photolysis_ids.push_back(reaction.id);
      if (has_electron(products) || has_positive_ion(products) ||
          has_negative_ion(products)) {
        report.selected_electron_impact_reactions++;
      }
      continue;
    }

    report.thermal_candidate_reactions++;
    if (charged) {
      report.charged_thermal_candidate_reactions++;
      if (has_electron(reactants) && has_positive_ion(reactants)) {
        report.dissociative_recombination_reactions++;
      } else {
        report.ion_mass_action_reactions++;
      }
    }
    if (reaction.reactant_ids.empty() || reaction.product_ids.empty() ||
        reaction.rate_blocks.empty()) {
      report.unsupported_reaction_ids.push_back(reaction.id);
    }
  }

  return report;
}

KBTitanReactionReport classify_kinetics_base_titan_reactions(
    std::string const& pun_path, std::string const& run_input_path) {
  KBTitanNetwork titan;
  titan.pun = parse_kinetics_base_pun(pun_path);
  titan.selection = parse_kinetics_base_run_input(run_input_path);
  return classify_kinetics_base_titan_reactions(titan);
}

void init_species_from_kinetics_base(std::string const& master_input_path) {
  auto data = parse_kinetics_base_master(master_input_path);

  species_names.clear();
  species_weights.clear();
  species_cref_R.clear();
  species_uref_R.clear();
  species_sref_R.clear();
  species_nasa9_low.clear();
  species_nasa9_high.clear();
  species_nasa9_Tmid.clear();

  for (auto const& sp : data.species) {
    species_names.push_back(sp.name);
    species_weights.push_back(sp.molecular_weight);
    species_cref_R.push_back(2.5);
    species_uref_R.push_back(0.0);
    species_sref_R.push_back(0.0);
    species_nasa9_low.push_back(sp.nasa9_low);
    species_nasa9_high.push_back(sp.nasa9_high);
    species_nasa9_Tmid.push_back(sp.nasa9_Tmid);
  }

  species_initialized = true;
}

static std::string format_equation(std::vector<std::string> const& reactants,
                                   std::vector<std::string> const& products,
                                   bool reversible) {
  std::string arrow = reversible ? " <=> " : " => ";
  // Count occurrences to produce stoich coefficients
  auto format_side = [](std::vector<std::string> const& species) {
    std::map<std::string, int> counts;
    std::vector<std::string> order;
    for (auto const& s : species) {
      if (counts.count(s) == 0) order.push_back(s);
      counts[s]++;
    }
    std::string result;
    for (size_t i = 0; i < order.size(); ++i) {
      if (i > 0) result += " + ";
      if (counts[order[i]] > 1)
        result += std::to_string(counts[order[i]]) + " ";
      result += order[i];
    }
    return result;
  };

  return format_side(reactants) + arrow + format_side(products);
}

static std::string collapse_whitespace(std::string const& s) {
  std::string result;
  bool last_space = false;
  for (char c : s) {
    if (std::isspace(c)) {
      if (!last_space) result += ' ';
      last_space = true;
    } else {
      result += std::toupper(c);
      last_space = false;
    }
  }
  while (!result.empty() && result.back() == ' ') result.pop_back();
  while (!result.empty() && result.front() == ' ') result.erase(result.begin());
  return result;
}

KineticsOptions kinetics_options_from_kinetics_base(
    std::string const& master_input_path, std::string const& photo_catalog_path,
    std::string const& cross_dir, bool verbose) {
  (void)photo_catalog_path;
  (void)cross_dir;
  auto master = parse_kinetics_base_master(master_input_path);
  init_species_from_kinetics_base(master_input_path);

  auto kinet = KineticsOptionsImpl::create();
  kinet->verbose(verbose);

  UnitSystem us;

  auto arrh = ArrheniusOptionsImpl::create();
  for (auto const& rxn : master.thermal) {
    if (rxn.has_M) continue;

    std::string eq = format_equation(rxn.reactants, rxn.products, true);
    arrh->reactions().push_back(Reaction(eq));

    double sum_stoich = 0.0;
    for (auto const& [_, coeff] : arrh->reactions().back().reactants()) {
      sum_stoich += coeff;
    }

    auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - sum_stoich,
                            -3. * (1. - sum_stoich));
    arrh->A().push_back(us.convert_from(rxn.A, unit));
    arrh->b().push_back(rxn.b);
    arrh->Ea_R().push_back(rxn.Ea_R);
    arrh->E4_R().push_back(0.0);
  }
  kinet->arrhenius() = arrh;

  auto tb = ThreeBodyOptionsImpl::create();
  for (auto const& rxn : master.thermal) {
    if (!rxn.has_M || rxn.has_kinf) continue;

    auto r = rxn.reactants;
    r.push_back("M");
    auto p = rxn.products;
    p.push_back("M");
    Reaction reaction(format_equation(r, p, true));

    double sum_stoich = 0.0;
    for (auto const& [name, coeff] : reaction.reactants()) {
      if (name != "M") sum_stoich += coeff;
    }
    sum_stoich += 1.0;

    auto unit = fmt::format("molecule^{} * cm^{} * s^-1", 1. - sum_stoich,
                            -3. * (1. - sum_stoich));

    tb->reactions().push_back(reaction);
    tb->k0_A().push_back(us.convert_from(rxn.A0, unit));
    tb->k0_b().push_back(rxn.b0);
    tb->k0_Ea_R().push_back(rxn.Ea_R0);
    tb->efficiencies().push_back({});
  }
  kinet->three_body() = tb;

  auto lf = LindemannFalloffOptionsImpl::create();
  for (auto const& rxn : master.thermal) {
    if (!rxn.has_M || !rxn.has_kinf) continue;

    auto r = rxn.reactants;
    r.push_back("M");
    auto p = rxn.products;
    p.push_back("M");
    Reaction reaction(format_equation(r, p, true));

    double sum_stoich = 0.0;
    for (auto const& [name, coeff] : reaction.reactants()) {
      if (name != "M") sum_stoich += coeff;
    }

    auto unit_low =
        fmt::format("molecule^{} * cm^{} * s^-1", 1. - (sum_stoich + 1.),
                    -3. * (1. - (sum_stoich + 1.)));
    auto unit_high = fmt::format("molecule^{} * cm^{} * s^-1", 1. - sum_stoich,
                                 -3. * (1. - sum_stoich));

    reaction.falloff_type("none");
    lf->reactions().push_back(reaction);
    lf->k0_A().push_back(us.convert_from(rxn.A0, unit_low));
    lf->k0_b().push_back(rxn.b0);
    lf->k0_Ea_R().push_back(rxn.Ea_R0);
    lf->kinf_A().push_back(us.convert_from(rxn.A_inf, unit_high));
    lf->kinf_b().push_back(rxn.b_inf);
    lf->kinf_Ea_R().push_back(rxn.Ea_R_inf);
    lf->efficiencies().push_back({});
  }
  kinet->lindemann_falloff() = lf;

  kinet->troe_falloff() = TroeFalloffOptionsImpl::create();
  kinet->sri_falloff() = SRIFalloffOptionsImpl::create();
  kinet->coagulation() = CoagulationOptionsImpl::create();
  kinet->evaporation() = EvaporationOptionsImpl::create();

  for (int id = 0; id < (int)species_names.size(); ++id) {
    kinet->vapor_ids().push_back(id);
  }
  std::sort(kinet->vapor_ids().begin(), kinet->vapor_ids().end());

  for (auto const& id : kinet->vapor_ids()) {
    kinet->cref_R().push_back(species_cref_R[id]);
    kinet->uref_R().push_back(species_uref_R[id]);
    kinet->sref_R().push_back(species_sref_R[id]);
    kinet->nasa9_low().push_back(species_nasa9_low[id]);
    kinet->nasa9_high().push_back(species_nasa9_high[id]);
    kinet->nasa9_Tmid().push_back(species_nasa9_Tmid[id]);
  }

  return kinet;
}

PhotoChemOptions photochem_options_from_kinetics_base(
    std::string const& master_input_path, std::string const& photo_catalog_path,
    std::string const& cross_dir, bool verbose) {
  auto master = parse_kinetics_base_master(master_input_path);

  init_species_from_kinetics_base(master_input_path);

  auto photo_chem = PhotoChemOptionsImpl::create();
  photo_chem->verbose(verbose);
  auto photo = PhotolysisOptionsImpl::create();

  std::map<std::string, std::string> catalog_map;
  std::map<std::string, KBCrossSectionFile> file_cache;
  std::map<std::string, KBCrossSectionFile const*> absorption_cache;

  auto interpolate_to_shared_wavelength = [&](std::vector<double> const& wl,
                                              std::vector<double> const& vals) {
    std::vector<double> out(photo->wavelength().size(), 0.0);
    if (wl.empty() || vals.empty()) return out;
    if (wl.size() == photo->wavelength().size()) {
      bool same_grid = true;
      for (size_t j = 0; j < wl.size(); ++j) {
        if (std::abs(photo->wavelength()[j] - wl[j]) >= 1e-12) {
          same_grid = false;
          break;
        }
      }
      if (same_grid) return vals;
    }

    for (size_t i = 0; i < photo->wavelength().size(); ++i) {
      double target = photo->wavelength()[i];
      if (target < wl.front() || target > wl.back()) continue;
      if (target == wl.front()) {
        out[i] = vals.front();
        continue;
      }
      if (target == wl.back()) {
        out[i] = vals.back();
        continue;
      }

      auto upper = std::lower_bound(wl.begin(), wl.end(), target);
      if (upper == wl.end()) {
        out[i] = vals.back();
        continue;
      }
      if (*upper == target) {
        out[i] = vals[upper - wl.begin()];
        continue;
      }

      auto lower = upper - 1;
      size_t j = lower - wl.begin();
      double frac = (target - wl[j]) / (wl[j + 1] - wl[j]);
      out[i] = vals[j] + frac * (vals[j + 1] - vals[j]);
    }
    return out;
  };

  auto interpolate_on_grid = [](std::vector<double> const& source_wl,
                                std::vector<double> const& source_vals,
                                double target) {
    if (source_wl.empty() || source_vals.empty()) return 0.0;
    if (target < source_wl.front() || target > source_wl.back()) return 0.0;
    if (target == source_wl.front()) return source_vals.front();
    if (target == source_wl.back()) return source_vals.back();

    auto upper = std::lower_bound(source_wl.begin(), source_wl.end(), target);
    if (upper == source_wl.end()) return source_vals.back();
    if (*upper == target) return source_vals[upper - source_wl.begin()];

    auto lower = upper - 1;
    size_t j = lower - source_wl.begin();
    double frac = (target - source_wl[j]) / (source_wl[j + 1] - source_wl[j]);
    return source_vals[j] + frac * (source_vals[j + 1] - source_vals[j]);
  };

  auto assign_or_check_temperature = [&](std::vector<double> const& temps) {
    if (temps.size() <= 1) return;
    if (photo->temperature().empty()) {
      photo->temperature() = temps;
      return;
    }
    TORCH_CHECK(photo->temperature() == temps,
                "All KINETICS-base photolysis reactions must use the same "
                "temperature grid");
  };

  auto find_absorption_dataset =
      [&](std::string const& parent_key,
          KBCrossSection const& branch_dataset) -> KBCrossSection const* {
    auto abs_it = absorption_cache.find(parent_key);
    if (abs_it == absorption_cache.end()) return nullptr;

    auto const& datasets = abs_it->second->datasets;
    KBCrossSection const* first_type0 = nullptr;
    for (auto const& abs_ds : datasets) {
      if (abs_ds.type != 0) continue;
      if (first_type0 == nullptr) first_type0 = &abs_ds;
      if (std::abs(abs_ds.temperature - branch_dataset.temperature) < 1e-12) {
        return &abs_ds;
      }
    }
    return first_type0;
  };

  if (!photo_catalog_path.empty()) {
    auto catalog = parse_kinetics_base_catalog(photo_catalog_path);

    auto build_side_key = [](std::vector<std::string> const& species) {
      std::map<std::string, int> counts;
      std::vector<std::string> order;
      for (auto const& s : species) {
        if (counts.count(s) == 0) order.push_back(s);
        counts[s]++;
      }
      std::string result;
      for (size_t i = 0; i < order.size(); ++i) {
        if (i > 0) result += "+";
        if (counts[order[i]] > 1) result += std::to_string(counts[order[i]]);
        result += to_upper(order[i]);
      }
      return result;
    };

    for (auto const& [cat_eq, fname] : catalog) {
      auto [cat_r, cat_p] = parse_equation_string(cat_eq);
      catalog_map[build_side_key(cat_r) + "=" + build_side_key(cat_p)] = fname;
    }

    for (auto const& [cat_eq, fname] : catalog) {
      if (file_cache.count(fname)) continue;
      std::string fpath = cross_dir.empty() ? fname : (cross_dir + "/" + fname);
      auto csf = parse_kinetics_base_cross_section(fpath);
      if (!csf.datasets.empty()) file_cache[fname] = std::move(csf);
    }

    for (auto const& [cat_eq, fname] : catalog) {
      auto it = file_cache.find(fname);
      if (it == file_cache.end()) continue;
      auto const& csf = it->second;
      auto has_type0 =
          std::any_of(csf.datasets.begin(), csf.datasets.end(),
                      [](KBCrossSection const& ds) { return ds.type == 0; });
      if (!has_type0) continue;
      auto [cat_r, cat_p] = parse_equation_string(cat_eq);
      absorption_cache[build_side_key(cat_r)] = &csf;
    }

    if (photo->wavelength().empty()) {
      std::set<double> shared_wavelengths;
      for (auto const& [_, csf] : file_cache) {
        for (auto const& ds : csf.datasets) {
          shared_wavelengths.insert(ds.wavelengths_nm.begin(),
                                    ds.wavelengths_nm.end());
        }
      }
      if (!shared_wavelengths.empty()) {
        photo->wavelength().assign(shared_wavelengths.begin(),
                                   shared_wavelengths.end());
      }
    }
  }

  TORCH_CHECK(
      master.photolysis.empty() || !photo_catalog_path.empty(),
      "PhotoChemOptions.from_kinetics_base requires a non-empty "
      "photo_catalog_path when the KINETICS-base mechanism contains "
      "photolysis reactions. Refusing to build photolysis reactions with an "
      "empty wavelength grid.");

  for (auto const& rxn : master.photolysis) {
    std::string eq = format_equation(rxn.reactants, rxn.products, false);
    photo->reactions().push_back(Reaction(eq));

    auto& reaction = photo->reactions().back();
    std::vector<std::string> branch_strs;
    std::string absorb_str;
    for (auto const& [sp, coeff] : reaction.reactants()) {
      absorb_str += sp + ":" + std::to_string((int)coeff) + " ";
    }
    branch_strs.push_back(absorb_str);

    if (reaction.products() != reaction.reactants()) {
      std::string prod_str;
      for (auto const& [sp, coeff] : reaction.products()) {
        prod_str += sp + ":" + std::to_string((int)coeff) + " ";
      }
      branch_strs.push_back(prod_str);
    }
    photo->branch_names().push_back(branch_strs);

    std::vector<Composition> branch_comps;
    for (auto const& s : branch_strs) {
      Composition comp;
      std::istringstream bss(s);
      std::string token;
      while (bss >> token) {
        auto colon = token.find(':');
        if (colon != std::string::npos) {
          comp[token.substr(0, colon)] = std::stod(token.substr(colon + 1));
        }
      }
      branch_comps.push_back(comp);
    }
    photo->branches().push_back(branch_comps);

    auto build_side_key2 = [](std::vector<std::string> const& species) {
      std::map<std::string, int> counts;
      std::vector<std::string> order;
      for (auto const& s : species) {
        if (counts.count(s) == 0) order.push_back(s);
        counts[s]++;
      }
      std::string result;
      for (size_t i = 0; i < order.size(); ++i) {
        if (i > 0) result += "+";
        if (counts[order[i]] > 1) result += std::to_string(counts[order[i]]);
        result += to_upper(order[i]);
      }
      return result;
    };

    std::string rxn_key =
        build_side_key2(rxn.reactants) + "=" + build_side_key2(rxn.products);
    int nbranch = std::max((int)branch_strs.size(), 1);
    auto cat_it = catalog_map.find(rxn_key);
    if (cat_it == catalog_map.end()) {
      TORCH_CHECK(!photo->wavelength().empty(),
                  "Missing photolysis catalog entry for reaction '", eq,
                  "' and no shared photolysis wavelength grid is available.");
      photo->cross_section_nslabs().push_back(1);
      photo->cross_section().insert(photo->cross_section().end(),
                                    photo->wavelength().size() * nbranch, 0.0);
      continue;
    }

    std::string const& fname = cat_it->second;
    std::string fpath = cross_dir.empty() ? fname : (cross_dir + "/" + fname);
    auto cache_it = file_cache.find(fname);
    if (cache_it == file_cache.end() || cache_it->second.datasets.empty()) {
      TORCH_CHECK(!photo->wavelength().empty(),
                  "Photolysis cross-section file '", fpath,
                  "' contains no usable datasets and no shared photolysis "
                  "wavelength grid is available for reaction '",
                  eq, "'.");
      photo->cross_section_nslabs().push_back(1);
      photo->cross_section().insert(photo->cross_section().end(),
                                    photo->wavelength().size() * nbranch, 0.0);
      continue;
    }
    auto const& csf = cache_it->second;

    std::vector<double> temps;
    temps.reserve(csf.datasets.size());
    for (auto const& ds : csf.datasets) temps.push_back(ds.temperature);
    if (csf.datasets.size() > 1) assign_or_check_temperature(temps);
    photo->cross_section_nslabs().push_back(csf.datasets.size());

    for (auto const& ds : csf.datasets) {
      TORCH_CHECK(
          ds.type == 0 || ds.type == 2,
          "Unsupported KINETICS-base cross-section dataset type ", ds.type,
          " in ", fpath,
          ". Only absorption (type 0) and branching/quantum-yield (type 2) "
          "are currently supported for direct photolysis reaction loading.");

      std::vector<double> vals =
          interpolate_to_shared_wavelength(ds.wavelengths_nm, ds.values);

      if (ds.type == 2) {
        std::string parent_key = build_side_key2(rxn.reactants);
        auto const* abs_ds = find_absorption_dataset(parent_key, ds);
        TORCH_CHECK(abs_ds != nullptr,
                    "Missing absorption cross-section (type 0) for photolysis "
                    "parent species ",
                    parent_key, " while loading ", fpath);
        for (size_t j = 0; j < vals.size(); ++j) {
          double abs_val = interpolate_on_grid(
              abs_ds->wavelengths_nm, abs_ds->values, photo->wavelength()[j]);
          vals[j] *= abs_val;
        }
      }

      for (double val : vals) {
        for (int bi = 0; bi < nbranch; ++bi) {
          photo->cross_section().push_back(val);
        }
      }
    }
  }

  photo_chem->photolysis() = photo;
  for (int id = 0; id < (int)species_names.size(); ++id) {
    photo_chem->vapor_ids().push_back(id);
  }
  for (auto const& id : photo_chem->vapor_ids()) {
    photo_chem->cref_R().push_back(species_cref_R[id]);
    photo_chem->uref_R().push_back(species_uref_R[id]);
    photo_chem->sref_R().push_back(species_sref_R[id]);
    photo_chem->nasa9_low().push_back(species_nasa9_low[id]);
    photo_chem->nasa9_high().push_back(species_nasa9_high[id]);
    photo_chem->nasa9_Tmid().push_back(species_nasa9_Tmid[id]);
  }

  return photo_chem;
}

}  // namespace kintera
