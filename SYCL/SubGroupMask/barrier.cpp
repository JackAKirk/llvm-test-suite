// REQUIRES: gpu, cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// Test that fails if sycl::ext::oneapi::group_barrier is not executed.

#include <limits>
#include <numeric>
#include <sycl.hpp>

using namespace sycl;

constexpr size_t G = 4;

template <typename T> void calc_b(T *a, T *b) {
  for (int i = 0; i < G; i++) {
    a[i] += i;
    b[i] += i;
  }

  for (int i = 0; i < G; i++) {

    b[0] += a[i] * b[i];
  }

  for (int i = 0; i < G; i++) {

    b[i] = b[0];
  }
};

template <typename T> void test(queue &Queue) {

  T *a;
  T *b;
  a = (T *)malloc(G * sizeof(T));
  b = (T *)malloc(G * sizeof(T));

  for (int i = 0; i < G; i++) {
    a[i] = 0;
    b[i] = 100;
  }

  auto d_a = sycl::malloc_device<T>(G, Queue);
  auto d_b = sycl::malloc_device<T>(G, Queue);

  Queue.memcpy(d_a, a, sizeof(T) * G).wait();
  Queue.memcpy(d_b, b, sizeof(T) * G).wait();

  nd_range<1> NdRange(G, G);

  Queue.submit([&](handler &cgh) {
    cgh.parallel_for(NdRange, [=](nd_item<1> NdItem) {
      ext::oneapi::sub_group SG = NdItem.get_sub_group();

      size_t i = NdItem.get_global_linear_id();

      auto mask =
          detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(
              0xffffffff, SG.get_max_local_range()[0]);

      d_a[i] += i;
      d_b[i] += i;

      if (i == 0) {
        for (int j = 0; j < G; j++)
          d_b[i] += d_a[j] * d_b[j];
      }

      sycl::ext::oneapi::group_barrier(SG, mask);

      d_a[i] = d_b[0];
    });
  });

  calc_b<T>(a, b);
  Queue.wait();

  Queue.memcpy(a, d_a, sizeof(T) * G).wait();

  for (int i = 0; i < G; i++) {
    assert(a[i] == b[i]);
  }
}

int main() {
  queue Queue;

  test<int>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
