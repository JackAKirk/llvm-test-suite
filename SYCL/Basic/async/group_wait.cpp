// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.run
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run
// RUN: env SYCL_DEVICE_FILTER=host %t.run
//
// Expected to crash on AMD
// XFAIL: hip_amd
//
// This test is designed to fail if wait_for(Group, E) breaks.

#include "common.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

template <typename T> class KernelName;

template <typename T> void initInputBuffer(buffer<T, 1> &Buf) {
  auto Acc = Buf.template get_access<access::mode::write>();
  const size_t NElems = Buf.get_count();

  for (size_t J = 0; J < NElems; J++)
    Acc[J] = (J < NElems / 2) ? J : 0;
}

template <typename T> int checkResults(buffer<T, 1> &OutBuf) {
  auto Out = OutBuf.template get_access<access::mode::read>();
  int EarlyFailout = 20;
  const size_t NElems = OutBuf.get_count();

  for (size_t J = 0; J < NElems; J++) {
    size_t ExpectedVal = (J < NElems / 2) ? (J * 2) : 0;
    if (!checkEqual(Out[J], ExpectedVal)) {
      std::cerr << std::string(typeid(T).name())
                << " : Incorrect value at index " << J
                << " : Expected: " << toString(ExpectedVal)
                << ", Computed: " << toString(Out[J]) << "\n";
      if (--EarlyFailout == 0)
        return 1;
    }
  }
  return EarlyFailout - 20;
}

template <typename T> int test() {

  queue Q;
  const size_t NElems =
      Q.get_device().get_info<info::device::max_work_group_size>();

  buffer<T, 1> InBuf(NElems);
  buffer<T, 1> OutBuf(NElems);

  initInputBuffer(InBuf);
  initOutputBuffer(OutBuf);

  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Out = OutBuf.template get_access<access::mode::write>(CGH);
     accessor<T, 1, access::mode::read_write, access::target::local> Local(
         range<1>{NElems}, CGH);

     nd_range<1> NDR{range<1>(NElems), range<1>(NElems)};
     CGH.parallel_for<KernelName<T>>(NDR, [=](nd_item<1> NDId) {
       Local[NDId.get_local_id()] = 1;

       auto Group = NDId.get_group();
       size_t NElemsToCopy = NElems / 2;

       {
         auto E = sycl::ext::oneapi::async_group_copy(
             Group, Local.get_pointer(), In.get_pointer(), NElemsToCopy);
         sycl::ext::oneapi::wait_for(Group, E);

         Local[NElems - NDId.get_local_id()] *= 2;
       }

       {
         auto E = sycl::ext::oneapi::async_group_copy(
             Group, Out.get_pointer(), Local.get_pointer(), NElemsToCopy);
         sycl::ext::oneapi::wait_for(Group, E);
       }
     });
   }).wait();

  return checkResults(OutBuf);
}

int main() {

  if (test<int>())
    return 1;
  if (test<uint>())
    return 1;
  if (test<double>())
    return 1;
  if (test<float>())
    return 1;
  if (test<long>())
    return 1;
  if (test<ulong>())
    return 1;
  if (test<vec<int, 1>>())
    return 1;
  if (test<int4>())
    return 1;

  std::cout << "Test passed.\n";
  return 0;
}
