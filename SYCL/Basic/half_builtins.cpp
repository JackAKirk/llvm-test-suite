// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// OpenCL CPU driver does not support cl_khr_fp16 extension
// UNSUPPORTED: cpu && opencl

#include <CL/sycl.hpp>

#include <cmath>
#include <unordered_set>

using namespace cl::sycl;

constexpr int N = 16 * 3; // divisible by all vector sizes

bool check(half a, half b) {
  return fabs(2 * (a - b) / (a + b)) <
             std::numeric_limits<cl::sycl::half>::epsilon() ||
         a < std::numeric_limits<cl::sycl::half>::min();
}

#define TEST_BUILTIN_1_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<half##SZ> a_buf((half##SZ *)&a[0], N / SZ);                         \
    buffer<half##SZ> d_buf((half##SZ *)&d[0], N / SZ);                         \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / SZ,                                                 \
                       [=](id<1> index) { D[index] = NAME(A[index]); });       \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(check(d[i], NAME(a[i])));                                           \
  }

// vectors of size 3 need separate test, as they actually have the size of 4
// halfs
#define TEST_BUILTIN_1_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    buffer<half3> a_buf((half3 *)&a[0], N / 4);                                \
    buffer<half3> d_buf((half3 *)&d[0], N / 4);                                \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / 4,                                                  \
                       [=](id<1> index) { D[index] = NAME(A[index]); });       \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    if (i % 4 != 3) {                                                          \
      assert(check(d[i], NAME(a[i])));                                         \
    }                                                                          \
  }

#define TEST_BUILTIN_1_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<half> a_buf(&a[0], N);                                              \
    buffer<half> d_buf(&d[0], N);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N, [=](id<1> index) { D[index] = NAME(A[index]); });    \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(check(d[i], NAME(a[i])));                                           \
  }

#define TEST_BUILTIN_1(NAME)                                                   \
  TEST_BUILTIN_1_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_1_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_2_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<half##SZ> a_buf((half##SZ *)&a[0], N / SZ);                         \
    buffer<half##SZ> b_buf((half##SZ *)&b[0], N / SZ);                         \
    buffer<half##SZ> d_buf((half##SZ *)&d[0], N / SZ);                         \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(                                                        \
          N / SZ, [=](id<1> index) { D[index] = NAME(A[index], B[index]); });  \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(check(d[i], NAME(a[i], b[i])));                                     \
  }

#define TEST_BUILTIN_2_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    buffer<half3> a_buf((half3 *)&a[0], N / 4);                                \
    buffer<half3> b_buf((half3 *)&b[0], N / 4);                                \
    buffer<half3> d_buf((half3 *)&d[0], N / 4);                                \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(                                                        \
          N / 4, [=](id<1> index) { D[index] = NAME(A[index], B[index]); });   \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    if (i % 4 != 3) {                                                          \
      assert(check(d[i], NAME(a[i], b[i])));                                   \
    }                                                                          \
  }

#define TEST_BUILTIN_2_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<half> a_buf(&a[0], N);                                              \
    buffer<half> b_buf(&b[0], N);                                              \
    buffer<half> d_buf(&d[0], N);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(                                                        \
          N, [=](id<1> index) { D[index] = NAME(A[index], B[index]); });       \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(check(d[i], NAME(a[i], b[i])));                                     \
  }

#define TEST_BUILTIN_2(NAME)                                                   \
  TEST_BUILTIN_2_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_2_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_3_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<half##SZ> a_buf((half##SZ *)&a[0], N / SZ);                         \
    buffer<half##SZ> b_buf((half##SZ *)&b[0], N / SZ);                         \
    buffer<half##SZ> c_buf((half##SZ *)&c[0], N / SZ);                         \
    buffer<half##SZ> d_buf((half##SZ *)&d[0], N / SZ);                         \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(check(d[i], NAME(a[i], b[i], c[i])));                               \
  }

#define TEST_BUILTIN_3_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    buffer<half3> a_buf((half3 *)&a[0], N / 4);                                \
    buffer<half3> b_buf((half3 *)&b[0], N / 4);                                \
    buffer<half3> c_buf((half3 *)&c[0], N / 4);                                \
    buffer<half3> d_buf((half3 *)&d[0], N / 4);                                \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / 4, [=](id<1> index) {                               \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    if (i % 4 != 3) {                                                          \
      assert(check(d[i], NAME(a[i], b[i], c[i])));                             \
    }                                                                          \
  }

#define TEST_BUILTIN_3_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<half> a_buf(&a[0], N);                                              \
    buffer<half> b_buf(&b[0], N);                                              \
    buffer<half> c_buf(&c[0], N);                                              \
    buffer<half> d_buf(&d[0], N);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(check(d[i], NAME(a[i], b[i], c[i])));                               \
  }

#define TEST_BUILTIN_3(NAME)                                                   \
  TEST_BUILTIN_3_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_3_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 16)

int main() {
  queue q;
  std::vector<half> a(N), b(N), c(N), d(N);
  for (int i = 0; i < N; i++) {
    a[i] = i / (half)N;
    b[i] = (N - i) / (half)N;
    c[i] = (half)(3 * i);
  }

  TEST_BUILTIN_1(fabs);
  TEST_BUILTIN_2(fmin);
  TEST_BUILTIN_2(fmax);
  TEST_BUILTIN_3(fma);

  return 0;
}
