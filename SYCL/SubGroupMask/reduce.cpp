// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Builtins in libclc are not implemented:
// XFAIL: hip_amd, cpu

// TODO: enable compile+runtime checks for operations defined in SPIR-V 1.3.
// That requires either adding a switch to clang (-spirv-max-version=1.3) or
// raising the spirv version from 1.1. to 1.3 for spirv translator
// unconditionally. Using operators specific for spirv 1.3 and higher with
// -spirv-max-version=1.1 being set by default causes assert/check fails
// in spirv translator.
// RUNx: %clangxx -fsycl -fsycl-targets=%sycl_triple -DSPIRV_1_3 %s -I . -o \
   %t13.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
using namespace sycl;

template <typename T, class BinaryOperation>
void test_impl(queue q,
          BinaryOperation binary_op,
          T identity) {
  constexpr int N = 128;
  size_t G = 64;
  
  std::array<T, N> input;
  std::iota(input.begin(), input.end(), sizeof(T));
  std::array<T, 4> output;
  T init = 42;
  
  {
    buffer<T> in_buf(input.data(), input.size());
    buffer<T> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            sycl::ext::oneapi::sub_group sg = it.get_sub_group();
			auto mask = detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(0xAAAA, sg.get_max_local_range()[0]);
			
            int lid = it.get_local_id(0);
            out[0] = reduce_over_group(sg, mask, in[lid], binary_op);
            out[1] = reduce_over_group(sg, mask, in[lid], init, binary_op);
            out[2] = joint_reduce(sg, mask, in.get_pointer(), in.get_pointer() + N,
                                  binary_op);
            out[3] = joint_reduce(sg, mask, in.get_pointer(), in.get_pointer() + N,
                                  init, binary_op);
          });
    });
  }
  
  T correct1=identity;
  for(int i=1;i<G;i+=2){
	  correct1 = binary_op(correct1, input[i]);
  }
  
  T correct2=identity;
  for(int i=0;i<N;i++){
	  correct2 = binary_op(correct2, input[i]);
  }
  
  assert(output[0] == correct1);
  assert(output[1] == binary_op(correct1, init));
  assert(output[2] == correct2);
  assert(output[3] == binary_op(correct2, init));
}

template <typename T>
void test(queue q) {
  test_impl<T>(q, sycl::plus<T>(), 0);
  test_impl<T>(q, sycl::minimum<T>(), std::numeric_limits<T>::max());
  test_impl<T>(q, sycl::maximum<T>(), std::numeric_limits<T>::lowest());
  test_impl<T>(q, sycl::multiplies<T>(), 1);
  test_impl<T>(q, sycl::bit_or<T>(), 0);
  test_impl<T>(q, sycl::bit_xor<T>(), 0);
  test_impl<T>(q, sycl::bit_and<T>(), ~0);
}


int main() {
  queue q;
  
  test<char>(q);
  test<unsigned char>(q);
  test<short>(q);
  test<unsigned short>(q);
  test<int>(q);
  test<unsigned int>(q);
  test<long>(q);
  test<unsigned>(q);

  std::cout << "Test passed." << std::endl;
}
