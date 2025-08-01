// C/C++
#include <stdexcept>

// snap
#include "func2.hpp"

std::unordered_map<std::string, int>& Func2Registrar::id_map() {
  static std::unordered_map<std::string, int> _id_map;
  return _id_map;
}

std::vector<user_func2>& Func2Registrar::host_func_list() {
  static std::vector<user_func2> _host_func_list;
  return _host_func_list;
}

user_func2 Func2Registrar::get_host_func(const std::string& name) {
  int id = get_id(name);
  if (id == -1) return nullptr;

  return host_func_list()[id];
}

int Func2Registrar::get_id(const std::string& name) {
  if (name.empty()) return -1;

  if (id_map().find(name) == id_map().end()) {
    throw std::runtime_error("Function " + name + " not registered.");
  }
  return id_map().at(name);
}

std::vector<user_func2> get_host_func2(std::vector<std::string> const& names) {
  std::vector<user_func2> funcs;
  for (const auto& name : names) {
    funcs.push_back(Func2Registrar::get_host_func(name));
  }
  return funcs;
}
