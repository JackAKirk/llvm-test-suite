// REQUIRES: gpu, cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 %s -o %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
using namespace sycl;

template <typename T, class BinaryOperation>
void test_impl(queue q, BinaryOperation binary_op, T identity) {
  constexpr int N = 7;
  size_t G = 32;

  std::array<T, N * 2> input;
  std::iota(input.begin(), input.end(), sizeof(T));
  std::array<T, 4> output;
  T init = 42;

  {
    buffer<T> in_buf(input.data(), input.size());
    buffer<T> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(nd_range<1>(G, G), [=](nd_item<1> it) {
        sycl::ext::oneapi::sub_group sg = it.get_sub_group();

        uint32_t msk = (1 << N) - 1;
        auto mask =
            detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(
                msk, sg.get_max_local_range()[0]);

        int lid = it.get_local_id(0);

        if (N - 1 < sg.get_local_id() && sg.get_local_id() < N * 2) {
          auto active_mask = it.ext_oneapi_active_sub_group_items();

          out[0] = reduce_over_group(sg, active_mask, in[lid], binary_op);
          out[1] = reduce_over_group(sg, active_mask, in[lid], init, binary_op);
        }
        out[0] =
            binary_op(out[0], reduce_over_group(sg, mask, in[lid], binary_op));
        out[1] =
            binary_op(out[1], reduce_over_group(sg, mask, in[lid], binary_op));

        auto active_mask = it.ext_oneapi_active_sub_group_items();
        out[2] = joint_reduce(sg, active_mask, in.get_pointer(),
                              in.get_pointer() + N, binary_op);
        out[3] = joint_reduce(sg, active_mask, in.get_pointer(),
                              in.get_pointer() + N, init, binary_op);
      });
    });
  }

  T correct1 = identity;
  for (int i = 0; i < N * 2; i += 1) {
    correct1 = binary_op(correct1, input[i]);
  }

  T correct2 = identity;
  for (int i = 0; i < N; i++) {
    correct2 = binary_op(correct2, input[i]);
  }

  assert(output[0] == correct1);
  assert(output[1] == binary_op(correct1, init));
  assert(output[2] == correct2);
  assert(output[3] == binary_op(correct2, init));
}

template <typename T> void test(queue q) {
  test_impl<T>(q, sycl::plus<T>(), 0);
  test_impl<T>(q, sycl::minimum<T>(), std::numeric_limits<T>::max());
  test_impl<T>(q, sycl::maximum<T>(), std::numeric_limits<T>::lowest());
}

int main() {
  queue q;

  test<int>(q);
  test<unsigned int>(q);
  test<short>(q);
  test<unsigned short>(q);
  test<char>(q);
  test<unsigned char>(q);

  std::cout << "Test passed." << std::endl;
}
