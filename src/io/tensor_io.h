#pragma once
// Binary tensor I/O matching the format written by python/dump_reference.py:
//   int32   num_tensors
//   repeat num_tensors times:
//     int32   name_len
//     char    name[name_len]      (not null-terminated)
//     int32   ndim
//     int32   shape[ndim]
//     float32 data[product(shape)]
#include <string>
#include <unordered_map>
#include <vector>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
};

std::unordered_map<std::string, Tensor> load_tensors(const std::string& path);
void save_tensors(const std::string& path, const std::unordered_map<std::string, Tensor>& tensors);
