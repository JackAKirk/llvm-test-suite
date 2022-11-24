// REQUIRES: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

// Checks that the sycl::ext::oneapi::experimental::cuda::__ldg builtins are
// returning the correct values.

#include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::cuda;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl;

template <typename T1> class KernelName;

template <typename T> bool checkEqual(vec<T, 2> A, vec<T, 2> B) {
  return A.x() == B.x() && A.y() == B.y();
}

template <typename T> bool checkEqual(vec<T, 4> A, vec<T, 4> B) {
  return A.x() == B.x() && A.y() == B.y() && A.z() == B.z() && A.w() == B.w();
}

template <typename T> void test(sycl::queue &q) {

  T a_loc;
  T b_loc;

  a_loc = 2;
  b_loc = 3;

  T *A = malloc_device<T>(1, q);
  T *B = malloc_device<T>(1, q);
  T *C = malloc_shared<T>(1, q);

  q.memcpy(A, &a_loc, sizeof(T));
  q.memcpy(B, &b_loc, sizeof(T));
  q.wait();

  q.submit([=](sycl::handler &h) {
    h.single_task<KernelName<T>>([=] {
      auto cacheA = __ldg(&A[0]);
      auto cacheB = __ldg(&B[0]);
      C[0] = cacheA + cacheB;
    });
  });

  q.wait();

  if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
    assert(C[0] == a_loc + b_loc);
  } else {
    assert(checkEqual(C[0], a_loc + b_loc));
  }

  free(A, q);
  free(B, q);
  free(C, q);

  return;
}

int main() {
  queue q;

  test<float>(q);
  test<double>(q);

  test<float2>(q);
  test<double2>(q);
  test<float4>(q);

  return 0;
}
