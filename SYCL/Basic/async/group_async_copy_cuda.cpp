// REQUIRES: cuda
// RUN: %clangxx -fsycl -std=c++17 -fsycl-targets=%sycl_triple %s -o %t.run
// RUN: %t.run
// RUN: %clangxx -fsycl -std=c++17 -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 %s -o %t.run
// TODO: Currently the CI does not have a sm_80 capable machine. Enable the test
// execution once it does.
// RUNx: %t.run

#include "common.h"

int main() {
  for (int Stride = 1; Stride < WorkGroupSize; Stride++) {
    if ((test<int>(Stride)) || (test<int2>(Stride)) || (test<int4>(Stride)) ||
        (test<uint>(Stride)) || (test<uint2>(Stride)) ||
        (test<uint4>(Stride)) || (test<double>(Stride)) ||
        (test<double2>(Stride)) || (test<float>(Stride)) ||
        (test<float2>(Stride)) || (test<float4>(Stride)) ||
        (test<long>(Stride)) || (test<long2>(Stride)) ||
        (test<ulong>(Stride)) || (test<ulong2>(Stride)) ||
        (test<char4>(Stride)) || (test<char8>(Stride)) ||
        (test<char16>(Stride)) || (test<schar4>(Stride)) ||
        (test<schar8>(Stride)) || (test<schar16>(Stride)) ||
        (test<uchar4>(Stride)) || (test<uchar8>(Stride)) ||
        (test<uchar16>(Stride)) || (test<short2>(Stride)) ||
        (test<short4>(Stride)) || (test<short8>(Stride)) ||
        (test<ushort2>(Stride)) || (test<ushort4>(Stride)) ||
        (test<ushort8>(Stride)) || (test<half2>(Stride)) ||
        (test<half4>(Stride)) || (test<half8>(Stride)) ||
        (test<vec<int, 1>>(Stride)) || (test<int4>(Stride)) ||
        (test<bool>(Stride)) || (test<vec<bool, 1>>(Stride)) ||
        (test<vec<bool, 4>>(Stride)) || (test<cl::sycl::cl_bool>(Stride)) ||
        (test<std::byte>(Stride)))
      return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}
