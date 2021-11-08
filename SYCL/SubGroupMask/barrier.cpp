// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Builtins in libclc are not implemented:
// XFAIL: hip_amd, cpu

#include <sycl.hpp>
#include <limits>
#include <numeric>

using namespace sycl;

template <typename T>
void check(queue &Queue, size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    std::vector<T> data(G);
    std::iota(data.begin(), data.end(), sizeof(T));
    buffer<T> resBuf(data.data(), range<1>(G));
    Queue.submit([&](handler &cgh) {
      auto resAcc = resBuf.template get_access<access::mode::read_write>(cgh);

      cgh.parallel_for(
          NdRange, [=](nd_item<1> NdItem) {
            ext::oneapi::sub_group SG = NdItem.get_sub_group();
			size_t g_id = NdItem.get_global_linear_id();
            size_t l_sg_id = SG.get_local_linear_id();
			size_t sg_size = SG.get_max_local_range()[0];
            size_t l_g_id = NdItem.get_local_linear_id();
            size_t SGoff = l_g_id - l_sg_id;
			auto mask = detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(~0, SG.get_max_local_range()[0]);

			if(l_sg_id==0){
				for(int i=SGoff;i<SGoff+sg_size;i++){
					resAcc[i] = 0;
				}
			}
			
			sycl::ext::oneapi::group_barrier(SG, mask);
			
			resAcc[l_g_id] += 1;
			
          });
    });
    auto resAcc = resBuf.template get_access<access::mode::read_write>();

    for (int i = 0; i < G; i++) {
		assert(resAcc[i]==1);
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

int main() {
  queue Queue;
  if (Queue.get_device().is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<int>(Queue);
  check<unsigned int>(Queue);
  check<long>(Queue);
  check<unsigned long>(Queue);
  check<float>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
