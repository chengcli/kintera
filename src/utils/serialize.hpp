#pragma once

// C/C++
#include <fstream>

// torch
#include <torch/torch.h>

namespace kintera {

void save_tensors(const std::map<std::string, torch::Tensor>& tensor_map,
                  const std::string& filename);

void load_tensors(std::map<std::string, torch::Tensor>& tensor_map,
                  const std::string& filename);

}  // namespace kintera
