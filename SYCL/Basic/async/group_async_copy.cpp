// RUN: %clangxx -fsycl -std=c++17 -fsycl-targets=%sycl_triple %s -o %t.run
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run
// RUN: env SYCL_DEVICE_FILTER=host %t.run

#include "common.hpp"

int main() {
  for (int Stride = 1; Stride < WorkGroupSize; Stride++) {
    if (test<int>(Stride))
      return 1;
    if (test<vec<int, 1>>(Stride))
      return 1;
    if (test<int4>(Stride))
      return 1;
    if (test<bool>(Stride))
      return 1;
    if (test<vec<bool, 1>>(Stride))
      return 1;
    if (test<vec<bool, 4>>(Stride))
      return 1;
    if (test<cl::sycl::cl_bool>(Stride))
      return 1;
    if (test<std::byte>(Stride))
      return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}
