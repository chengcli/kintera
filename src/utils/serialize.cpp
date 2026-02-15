// torch
#include <torch/script.h>
#include <torch/serialize.h>

// kintera
#include "serialize.hpp"

namespace kintera {

void save_tensors(const std::map<std::string, torch::Tensor>& tensor_map,
                  const std::string& filename) {
  torch::serialize::OutputArchive archive;
  for (const auto& pair : tensor_map) {
    archive.write(pair.first, pair.second);
  }
  archive.save_to(filename);
}

std::map<std::string, torch::Tensor> load_tensors(const std::string& filename) {
  std::map<std::string, torch::Tensor> data;

  // get keys
  torch::jit::Module m = torch::jit::load(filename);

  for (const auto& p : m.named_parameters(/*recurse=*/true)) {
    data[p.name] = p.value;
  }

  for (const auto& p : m.named_buffers(/*recurse=*/true)) {
    data[p.name] = p.value;
  }

  /*torch::serialize::InputArchive archive;
  archive.load_from(filename);
  for (auto& pair : data) {
    try {
      archive.read(pair.first, pair.second);
    } catch (const c10::Error& e) {
      // skip missing tensors
    }
  }*/

  return data;
}

}  // namespace kintera
