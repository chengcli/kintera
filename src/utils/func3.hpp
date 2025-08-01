#pragma once

// C/C++
#include <string>
#include <unordered_map>
#include <vector>

using user_func3 = double (*)(double, double, double);

class Func3Registrar {
 public:
  Func3Registrar(const std::string& name, user_func3 func) {
    int id = static_cast<int>(host_func_list().size());
    id_map()[name] = id;
    host_func_list().push_back(func);
  }

  static user_func3 get_host_func(const std::string& name);
  static int get_id(const std::string& name);

  static std::vector<std::string> list_names() {
    std::vector<std::string> names;
    for (const auto& kv : id_map()) names.push_back(kv.first);
    return names;
  }

 private:
  static std::unordered_map<std::string, int>& id_map();
  static std::vector<user_func3>& host_func_list();
};

std::vector<user_func3> get_host_func3(std::vector<std::string> const& names);

#ifdef __CUDACC__
#include <thrust/device_vector.h>

thrust::device_vector<user_func3> get_device_func3(
    std::vector<std::string> const& names);

#endif
