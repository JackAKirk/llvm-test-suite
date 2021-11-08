// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
using namespace sycl;

int main() {
  queue q;
  
  constexpr int N = 128;
  
  std::array<uint32_t, N> output;
  int sg_size;
  
  {
    buffer<uint32_t> out_buf(output.data(), output.size());
    buffer<int> sg_size_buf(&sg_size, 1);

    q.submit([&](handler &cgh) {
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      accessor sg_size_acc{sg_size_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(
          nd_range<1>(N, N), [=](nd_item<1> it) {
			auto mask = it.ext_oneapi_active_sub_group_items();
			uint32_t mask_bits;
			mask.extract_bits(mask_bits);
			
			out[it.get_global_linear_id()] = mask_bits;
			sg_size_acc[0] = it.get_sub_group().get_local_linear_range();
          });
    });
  }
  for(int i=0;i<N;i++){
	//each mask must have the bit for the calling work items set
	assert(output[i] & (1 << (i % sg_size)));
  }
  
  std::cout << "Test passed." << std::endl;
}

