// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend --cuda-gpu-arch=sm_80
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Only cuda backend implements bf16
// REQUIRES: cuda

#include <CL/sycl.hpp>

#include <cmath>
#include <unordered_set>

using namespace cl::sycl;
using namespace cl::sycl::ext::oneapi;

constexpr int N = 16 * 3; // divisible by all vector sizes
constexpr float bf16_eps = 0.00390625;

union conv {
  float f;
  vec<uint16_t, 2> u;
  uint32_t u2;
};

uint16_t to_bf16(float x) {
  conv c;
  c.f = x;
  return c.u.y();
}

uint32_t to_bf16x2(float x, float y) {
  conv c1;
  c1.f = x;
  conv c2;
  c2.f = y;
  conv c3;
  c3.u.x() = c1.u.y();
  c3.u.y() = c2.u.y();
  return c3.u2;
}

float from_bf16(uint16_t x) {
  conv c;
  c.u.y() = x;
  c.u.x() = 0;
  return c.f;
}

float2 from_bf16x2(uint32_t x) {
  conv c;
  c.u2 = x;
  conv c2;
  c2.u.x() = 0;
  c2.u.y() = c.u.x();
  float2 res;
  res.x() = c2.f;
  c2.u.y() = c.u.y();
  res.y() = c2.f;
  return res;
}

bool check(float a, float b) {
  return fabs(2 * (a - b) / (a + b)) > bf16_eps * 2;
}

#define TEST_BUILTIN_1_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<float##SZ> a_buf((float##SZ *)&a[0], N / SZ);                       \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        vec<uint16_t, SZ> arg;                                                 \
        for (int i = 0; i < SZ; i++) {                                         \
          arg[i] = to_bf16(A[index][i]);                                       \
        }                                                                      \
        vec<uint16_t, SZ> res = NAME(arg);                                     \
        for (int i = 0; i < SZ; i++) {                                         \
          if (check(from_bf16(res[i]), NAME(A[index][i]))) {                   \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          vec<uint32_t, SZ> x2_arg;                                            \
          for (int i = 0; i < SZ; i++) {                                       \
            x2_arg[i] = to_bf16x2(A[index][i], A[index + 1][i]);               \
          }                                                                    \
          vec<uint32_t, SZ> x2_res = NAME(x2_arg);                             \
          for (int i = 0; i < SZ; i++) {                                       \
            float2 res2 = from_bf16x2(x2_res[i]);                              \
            if (check(res2.x(), NAME(A[index][i])) ||                          \
                check(res2.y(), NAME(A[index + 1][i]))) {                      \
              ERR[0] = 1;                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

// vectors of size 3 need separate test, as they actually have the size of 4
// floats
#define TEST_BUILTIN_1_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float3> a_buf((float3 *)&a[0], N / 4);                              \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N / 4, [=](id<1> index) {                               \
        vec<uint16_t, 3> arg;                                                  \
        for (int i = 0; i < 3; i++) {                                          \
          arg[i] = to_bf16(A[index][i]);                                       \
        }                                                                      \
        vec<uint16_t, 3> res = NAME(arg);                                      \
        for (int i = 0; i < 3; i++) {                                          \
          if (check(from_bf16(res[i]), NAME(A[index][i]))) {                   \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          vec<uint32_t, 3> x2_arg;                                             \
          for (int i = 0; i < 3; i++) {                                        \
            x2_arg[i] = to_bf16x2(A[index][i], A[index + 1][i]);               \
          }                                                                    \
          vec<uint32_t, 3> x2_res = NAME(x2_arg);                              \
          for (int i = 0; i < 3; i++) {                                        \
            float2 res2 = from_bf16x2(x2_res[i]);                              \
            if (check(res2.x(), NAME(A[index][i])) ||                          \
                check(res2.y(), NAME(A[index + 1][i]))) {                      \
              ERR[0] = 1;                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_1_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(from_bf16(NAME(to_bf16(A[index]))), NAME(A[index]))) {       \
          ERR[0] = 1;                                                          \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          float2 res = from_bf16x2(NAME(to_bf16x2(A[index], A[index + 1])));   \
          if (check(res.x(), NAME(A[index])) ||                                \
              check(res.y(), NAME(A[index + 1]))) {                            \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_1(NAME)                                                   \
  TEST_BUILTIN_1_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_1_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_2_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<float##SZ> a_buf((float##SZ *)&a[0], N / SZ);                       \
    buffer<float##SZ> b_buf((float##SZ *)&b[0], N / SZ);                       \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        vec<uint16_t, SZ> arg0, arg1;                                          \
        for (int i = 0; i < SZ; i++) {                                         \
          arg0[i] = to_bf16(A[index][i]);                                      \
          arg1[i] = to_bf16(B[index][i]);                                      \
        }                                                                      \
        vec<uint16_t, SZ> res = NAME(arg0, arg1);                              \
        for (int i = 0; i < SZ; i++) {                                         \
          if (check(from_bf16(res[i]), NAME(A[index][i], B[index][i]))) {      \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          vec<uint32_t, SZ> x2_arg0, x2_arg1;                                  \
          for (int i = 0; i < SZ; i++) {                                       \
            x2_arg0[i] = to_bf16x2(A[index][i], A[index + 1][i]);              \
            x2_arg1[i] = to_bf16x2(B[index][i], B[index + 1][i]);              \
          }                                                                    \
          vec<uint32_t, SZ> x2_res = NAME(x2_arg0, x2_arg1);                   \
          for (int i = 0; i < SZ; i++) {                                       \
            float2 res2 = from_bf16x2(x2_res[i]);                              \
            if (check(res2.x(), NAME(A[index][i], B[index][i])) ||             \
                check(res2.y(), NAME(A[index + 1][i], B[index + 1][i]))) {     \
              ERR[0] = 1;                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_2_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float3> a_buf((float3 *)&a[0], N / 4);                              \
    buffer<float3> b_buf((float3 *)&b[0], N / 4);                              \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N / 4, [=](id<1> index) {                               \
        vec<uint16_t, 3> arg0, arg1;                                           \
        for (int i = 0; i < 3; i++) {                                          \
          arg0[i] = to_bf16(A[index][i]);                                      \
          arg1[i] = to_bf16(B[index][i]);                                      \
        }                                                                      \
        vec<uint16_t, 3> res = NAME(arg0, arg1);                               \
        for (int i = 0; i < 3; i++) {                                          \
          if (check(from_bf16(res[i]), NAME(A[index][i], B[index][i]))) {      \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          vec<uint32_t, 3> x2_arg0, x2_arg1;                                   \
          for (int i = 0; i < 3; i++) {                                        \
            x2_arg0[i] = to_bf16x2(A[index][i], A[index + 1][i]);              \
            x2_arg1[i] = to_bf16x2(B[index][i], B[index + 1][i]);              \
          }                                                                    \
          vec<uint32_t, 3> x2_res = NAME(x2_arg0, x2_arg1);                    \
          for (int i = 0; i < 3; i++) {                                        \
            float2 res2 = from_bf16x2(x2_res[i]);                              \
            if (check(res2.x(), NAME(A[index][i], B[index][i])) ||             \
                check(res2.y(), NAME(A[index + 1][i], B[index + 1][i]))) {     \
              ERR[0] = 1;                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_2_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<float> b_buf(&b[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(from_bf16(NAME(to_bf16(A[index]), to_bf16(B[index]))),       \
                  NAME(A[index], B[index]))) {                                 \
          ERR[0] = 1;                                                          \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          float2 res = from_bf16x2(NAME(to_bf16x2(A[index], A[index + 1]),     \
                                        to_bf16x2(B[index], B[index + 1])));   \
          if (check(res.x(), NAME(A[index], B[index])) ||                      \
              check(res.y(), NAME(A[index + 1], B[index + 1]))) {              \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_2(NAME)                                                   \
  TEST_BUILTIN_2_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_2_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_3_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<float##SZ> a_buf((float##SZ *)&a[0], N / SZ);                       \
    buffer<float##SZ> b_buf((float##SZ *)&b[0], N / SZ);                       \
    buffer<float##SZ> c_buf((float##SZ *)&c[0], N / SZ);                       \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        vec<uint16_t, SZ> arg0, arg1, arg2;                                    \
        for (int i = 0; i < SZ; i++) {                                         \
          arg0[i] = to_bf16(A[index][i]);                                      \
          arg1[i] = to_bf16(B[index][i]);                                      \
          arg2[i] = to_bf16(C[index][i]);                                      \
        }                                                                      \
        vec<uint16_t, SZ> res = NAME(arg0, arg1, arg2);                        \
        for (int i = 0; i < SZ; i++) {                                         \
          if (check(from_bf16(res[i]),                                         \
                    NAME(A[index][i], B[index][i], C[index][i]))) {            \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          vec<uint32_t, SZ> x2_arg0, x2_arg1, x2_arg2;                         \
          for (int i = 0; i < SZ; i++) {                                       \
            x2_arg0[i] = to_bf16x2(A[index][i], A[index + 1][i]);              \
            x2_arg1[i] = to_bf16x2(B[index][i], B[index + 1][i]);              \
            x2_arg2[i] = to_bf16x2(C[index][i], C[index + 1][i]);              \
          }                                                                    \
          vec<uint32_t, SZ> x2_res = NAME(x2_arg0, x2_arg1, x2_arg2);          \
          for (int i = 0; i < SZ; i++) {                                       \
            float2 res2 = from_bf16x2(x2_res[i]);                              \
            if (check(res2.x(),                                                \
                      NAME(A[index][i], B[index][i], C[index][i])) ||          \
                check(res2.y(), NAME(A[index + 1][i], B[index + 1][i],         \
                                     C[index + 1][i]))) {                      \
              ERR[0] = 1;                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_3_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float3> a_buf((float3 *)&a[0], N / 4);                              \
    buffer<float3> b_buf((float3 *)&b[0], N / 4);                              \
    buffer<float3> c_buf((float3 *)&c[0], N / 4);                              \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N / 4, [=](id<1> index) {                               \
        vec<uint16_t, 3> arg0, arg1, arg2;                                     \
        for (int i = 0; i < 3; i++) {                                          \
          arg0[i] = to_bf16(A[index][i]);                                      \
          arg1[i] = to_bf16(B[index][i]);                                      \
          arg2[i] = to_bf16(C[index][i]);                                      \
        }                                                                      \
        vec<uint16_t, 3> res = NAME(arg0, arg1, arg2);                         \
        for (int i = 0; i < 3; i++) {                                          \
          if (check(from_bf16(res[i]),                                         \
                    NAME(A[index][i], B[index][i], C[index][i]))) {            \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          vec<uint32_t, 3> x2_arg0, x2_arg1, x2_arg2;                          \
          for (int i = 0; i < 3; i++) {                                        \
            x2_arg0[i] = to_bf16x2(A[index][i], A[index + 1][i]);              \
            x2_arg1[i] = to_bf16x2(B[index][i], B[index + 1][i]);              \
            x2_arg2[i] = to_bf16x2(C[index][i], C[index + 1][i]);              \
          }                                                                    \
          vec<uint32_t, 3> x2_res = NAME(x2_arg0, x2_arg1, x2_arg2);           \
          for (int i = 0; i < 3; i++) {                                        \
            float2 res2 = from_bf16x2(x2_res[i]);                              \
            if (check(res2.x(),                                                \
                      NAME(A[index][i], B[index][i], C[index][i])) ||          \
                check(res2.y(), NAME(A[index + 1][i], B[index + 1][i],         \
                                     C[index + 1][i]))) {                      \
              ERR[0] = 1;                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_3_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<float> b_buf(&b[0], N);                                             \
    buffer<float> c_buf(&c[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(from_bf16(NAME(to_bf16(A[index]), to_bf16(B[index]),         \
                                 to_bf16(C[index]))),                          \
                  NAME(A[index], B[index], C[index]))) {                       \
          ERR[0] = 1;                                                          \
        }                                                                      \
        if (index % 2 == 0) {                                                  \
          float2 res = from_bf16x2(NAME(to_bf16x2(A[index], A[index + 1]),     \
                                        to_bf16x2(B[index], B[index + 1]),     \
                                        to_bf16x2(C[index], C[index + 1])));   \
          if (check(res.x(), NAME(A[index], B[index], C[index])) ||            \
              check(res.y(),                                                   \
                    NAME(A[index + 1], B[index + 1], C[index + 1]))) {         \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_3(NAME)                                                   \
  TEST_BUILTIN_3_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_3_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 16)

int main() {
  queue q;
  std::vector<float> a(N), b(N), c(N);
  int err = 0;
  for (int i = 0; i < N; i++) {
    a[i] = (i - N / 2) / (float)N;
    b[i] = (N / 2 - i) / (float)N;
    c[i] = (float)(3 * i);
  }

  TEST_BUILTIN_1(fabs);
  TEST_BUILTIN_2(fmin);
  TEST_BUILTIN_2(fmax);
  TEST_BUILTIN_3(fma);

  return 0;
}
