// REQUIRES: gpu, cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

// Tests that inter device copy (P2P) works correctly when the copy is made
// across distinct contexts for buffers.  When a pair of command groups are
// submitted to different queues with distinct contexts the runtime copies
// buffer memory across contexts. For the cuda backend this copy is made
// directly across devices.

#include "utils.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>
#include <iostream>

using namespace cl::sycl;

constexpr long width = 16;
constexpr long height = 5;
constexpr long total = width * height;

constexpr long depth = 3;
constexpr long total3D = total * depth;

void copyP2P_1D() {

  std::vector<float> data_1D(width, 13);

  {
    std::cout << "-- 1D" << std::endl;
    buffer<float, 1> buffer_1D(data_1D.data(), range<1>(width));
    auto Queues = getQueues();

    Queues[0].submit([&](handler &cgh) {
      auto readwrite = buffer_1D.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyH2D_1D>(
          buffer_1D.get_range(),
          [=](id<1> index) { readwrite[index] = readwrite[index] * -1; });
    });

    Queues[1].submit([&](handler &cgh) {
      auto readwrite = buffer_1D.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyH2D_1D_2nd>(
          buffer_1D.get_range(),
          [=](id<1> index) { readwrite[index] = readwrite[index] * 10; });
    });
    const auto host_accessor =
        buffer_1D.get_access<cl::sycl::access::mode::read>();
    for (int i = 0; i < width; i++) {
      assert(host_accessor[i] == -130);
    }
  }
  std::cout << "about to destruct 1D" << std::endl;
}

void copyP2P_2D() {

  std::vector<float> data_2D(total, 7);
  {
    std::cout << "-- 2D" << std::endl;
    buffer<float, 2> buffer_2D(data_2D.data(), range<2>(height, width));
    std::vector<sycl::queue> Queues = getQueues();

    Queues[0].submit([&](handler &cgh) {
      auto readwrite = buffer_2D.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyH2D_2D>(
          buffer_2D.get_range(),
          [=](id<2> index) { readwrite[index] = readwrite[index] * -1; });
    });

    Queues[1].submit([&](handler &cgh) {
      auto readwrite = buffer_2D.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyH2D_2D_2nd>(
          buffer_2D.get_range(),
          [=](id<2> index) { readwrite[index] = readwrite[index] * 10; });
    });

    const auto host_accessor =
        buffer_2D.get_access<cl::sycl::access::mode::read>();

    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        assert(host_accessor[i][j] == -70);
  }
  std::cout << "about to destruct 2D" << std::endl;
}

void copyP2P_3D() {

  std::vector<float> data_3D(total3D, 17);
  {
    std::cout << "-- 3D" << std::endl;
    buffer<float, 3> buffer_3D(data_3D.data(), range<3>(depth, height, width));
    std::vector<sycl::queue> Queues = getQueues();

    Queues[0].submit([&](handler &cgh) {
      auto readwrite = buffer_3D.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class copyH2D_3D>(
          buffer_3D.get_range(),
          [=](id<3> index) { readwrite[index] = readwrite[index] * -1; });
    });

    Queues[1].submit([&](handler &cgh) {
      auto readwrite = buffer_3D.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyH2D_3D_2nd>(
          buffer_3D.get_range(),
          [=](id<3> index) { readwrite[index] = readwrite[index] * 10; });
    });

    const auto host_accessor =
        buffer_3D.get_access<cl::sycl::access::mode::read>();

    for (int i = 0; i < depth; i++)
      for (int j = 0; j < height; j++)
        for (int k = 0; k < width; k++)
          assert(host_accessor[i][j][k] == -170);
  }
  std::cout << "about to destruct 3D" << std::endl;
}

int main() {
  remind(width, height, depth);
  for (int i = 0; i < 5; i++) {
    copyP2P_1D();
    copyP2P_2D();
    copyP2P_3D();
  }
}

// CHECK-LABEL: -- 1D
// CHECK: ---> piextEnqueueMemBufferCopyPeer(
// CHECK: <unknown> : 64
// CHECK: about to destruct 1D
// CHECK-LABEL: -- 2D
// CHECK: ---> piextEnqueueMemBufferCopyRectPeer(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK: about to destruct 2D
// CHECK-LABEL: -- 3D
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK: about to destruct 3D
