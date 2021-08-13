// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

#include "remind_utils.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>
#include <iostream>
using namespace cl::sycl;

constexpr long width = 16;
constexpr long height = 5;
constexpr long total = width * height;

constexpr long depth = 3;
constexpr long total3D = total * depth;

void testcopyH2DBuffer() {
  // copy between two queues triggers a piEnqueueMemBufferMap followed by
  // copyH2D, followed by a copyD2H, followed by a piEnqueueMemUnmap
  // Here we only care about checking copyH2D

  std::cout << "start copyH2D-buffer" << std::endl;
  std::vector<float> data_from_1D(width, 13);
  std::vector<float> data_to_1D(width, 0);
  std::vector<float> data_from_2D(total, 7);
  std::vector<float> data_to_2D(total, 0);
  std::vector<float> data_from_3D(total3D, 17);
  std::vector<float> data_to_3D(total3D, 0);

  {
    buffer<float, 1> buffer_from_1D(data_from_1D.data(), range<1>(width));
    buffer<float, 1> buffer_to_1D(data_to_1D.data(), range<1>(width));
    queue myQueue;
    queue otherQueue;
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_1D>(
          buffer_from_1D.get_range(),
          [=](id<1> index) { write[index] = read[index] * -1; });
    });
    myQueue.wait();

    otherQueue.submit([&](handler &cgh) {
      auto read = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_1D_2nd>(
          buffer_from_1D.get_range(),
          [=](id<1> index) { write[index] = read[index] * 10; });
    });
  } // ~buffer 1D

  {
    buffer<float, 2> buffer_from_2D(data_from_2D.data(),
                                    range<2>(height, width));
    buffer<float, 2> buffer_to_2D(data_to_2D.data(), range<2>(height, width));
    queue myQueue;
    queue otherQueue;
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_2D>(
          buffer_from_2D.get_range(),
          [=](id<2> index) { write[index] = read[index] * -1; });
    });

    otherQueue.submit([&](handler &cgh) {
      auto read = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_2D_2nd>(
          buffer_from_2D.get_range(),
          [=](id<2> index) { write[index] = read[index] * 10; });
    });
  } // ~buffer 2D

  {
    buffer<float, 3> buffer_from_3D(data_from_3D.data(),
                                    range<3>(depth, height, width));
    buffer<float, 3> buffer_to_3D(data_to_3D.data(),
                                  range<3>(depth, height, width));
    queue myQueue;
    queue otherQueue;
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_3D>(
          buffer_from_3D.get_range(),
          [=](id<3> index) { write[index] = read[index] * -1; });
    });

    otherQueue.submit([&](handler &cgh) {
      auto read = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_3D_2nd>(
          buffer_from_3D.get_range(),
          [=](id<3> index) { write[index] = read[index] * 10; });
    });
  } // ~buffer 3D

  std::cout << "end copyH2D-buffer" << std::endl;
}

// --------------

int main() {
  remind(width, height, depth);
  testcopyH2DBuffer();
}

// CHECK-LABEL: start copyH2D-buffer
// CHECK: ---> piEnqueueMemBufferWrite(
// CHECK: <unknown> : 64
// CHECK:  ---> piEnqueueMemBufferWriteRect(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 0
// CHECK-NEXT: <unknown> : 64
// CHECK:  ---> piEnqueueMemBufferWriteRect(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK: end copyH2D-buffer
