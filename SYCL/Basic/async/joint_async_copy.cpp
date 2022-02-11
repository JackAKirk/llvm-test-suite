// RUN: %clangxx -fsycl -std=c++17 -fsycl-targets=%sycl_triple %s -o %t.run
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run
// RUN: env SYCL_DEVICE_FILTER=host %t.run
//
// Crashes on AMD
// XFAIL: hip_amd

#include "common.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;
using sycl::ext::oneapi::experimental::dest_stride;
using sycl::ext::oneapi::experimental::src_stride;

template <typename T, typename G> class TypeHelper;

template <typename T, typename G>
using KernelName = class TypeHelper<
    typename std::conditional<std::is_same<T, std::byte>::value, unsigned char,
                              T>::type,
    G>;

template <typename T, typename G> int test(size_t Stride, queue &Q) {

  buffer<T, 1> InBuf(NElems);
  buffer<T, 1> OutBuf(NElems);

  initInputBuffer(InBuf, Stride);
  initOutputBuffer(OutBuf);

  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Out = OutBuf.template get_access<access::mode::write>(CGH);
     accessor<T, 1, access::mode::read_write, access::target::local> Local(
         range<1>{WorkGroupSize}, CGH);

     nd_range<1> NDR{range<1>(NElems), range<1>(WorkGroupSize)};
     CGH.parallel_for<KernelName<T, G>>(NDR, [=](nd_item<1> NDId) {
       auto GrId = NDId.get_group_linear_id();
       size_t NElemsToCopy =
           WorkGroupSize / Stride + ((WorkGroupSize % Stride) ? 1 : 0);
       size_t Offset = GrId * WorkGroupSize;

       auto Group = [&NDId]() {
         if constexpr (std::is_same<G, sub_group>::value) {
           return NDId.get_sub_group();
         } else {
           return NDId.get_group();
         }
       }();

       if (Stride == 1) { // Check the version without stride arg.
         auto E = sycl::ext::oneapi::experimental::joint_async_copy(
             Group, In.get_pointer() + Offset, Local.get_pointer(),
             NElemsToCopy);
         sycl::ext::oneapi::experimental::wait_for(Group, E);
       } else {
         auto E = sycl::ext::oneapi::experimental::joint_async_copy(
             Group, In.get_pointer() + Offset, Local.get_pointer(),
             NElemsToCopy, src_stride{Stride});
         sycl::ext::oneapi::experimental::wait_for(Group, E);
       }

       if (Stride == 1) { // Check the version without stride arg.
         auto E = sycl::ext::oneapi::experimental::joint_async_copy(
             Group, Local.get_pointer(), Out.get_pointer() + Offset,
             NElemsToCopy);
         sycl::ext::oneapi::experimental::wait_for(Group, E);
       } else {
         auto E = sycl::ext::oneapi::experimental::joint_async_copy(
             Group, Local.get_pointer(), Out.get_pointer() + Offset,
             NElemsToCopy, dest_stride{Stride});
         sycl::ext::oneapi::experimental::wait_for(Group, E);
       }
     });
   }).wait();

  return checkResults(OutBuf, Stride);
}

int main() {

  queue Q;

  for (int Stride = 1; Stride < WorkGroupSize; Stride++) {
    if (test<int, group<1>>(Stride, Q))
      return 1;
    if (test<int2, group<1>>(Stride, Q))
      return 1;
    if (test<int4, group<1>>(Stride, Q))
      return 1;
    if (test<uint, group<1>>(Stride, Q))
      return 1;
    if (test<uint2, group<1>>(Stride, Q))
      return 1;
    if (test<uint4, group<1>>(Stride, Q))
      return 1;
    if (test<double, group<1>>(Stride, Q))
      return 1;
    if (test<double2, group<1>>(Stride, Q))
      return 1;
    if (test<float, group<1>>(Stride, Q))
      return 1;
    if (test<float2, group<1>>(Stride, Q))
      return 1;
    if (test<float4, group<1>>(Stride, Q))
      return 1;
    if (test<long, group<1>>(Stride, Q))
      return 1;
    if (test<long2, group<1>>(Stride, Q))
      return 1;
    if (test<ulong, group<1>>(Stride, Q))
      return 1;
    if (test<ulong2, group<1>>(Stride, Q))
      return 1;
    if (test<char4, group<1>>(Stride, Q))
      return 1;
    if (test<char8, group<1>>(Stride, Q))
      return 1;
    if (test<char16, group<1>>(Stride, Q))
      return 1;
    if (test<schar4, group<1>>(Stride, Q))
      return 1;
    if (test<schar8, group<1>>(Stride, Q))
      return 1;
    if (test<schar16, group<1>>(Stride, Q))
      return 1;
    if (test<uchar4, group<1>>(Stride, Q))
      return 1;
    if (test<uchar8, group<1>>(Stride, Q))
      return 1;
    if (test<uchar16, group<1>>(Stride, Q))
      return 1;
    if (test<short2, group<1>>(Stride, Q))
      return 1;
    if (test<short4, group<1>>(Stride, Q))
      return 1;
    if (test<short8, group<1>>(Stride, Q))
      return 1;
    if (test<ushort2, group<1>>(Stride, Q))
      return 1;
    if (test<ushort4, group<1>>(Stride, Q))
      return 1;
    if (test<ushort8, group<1>>(Stride, Q))
      return 1;
    if (test<half2, group<1>>(Stride, Q))
      return 1;
    if (test<half4, group<1>>(Stride, Q))
      return 1;
    if (test<half8, group<1>>(Stride, Q))
      return 1;

    if (test<vec<int, 1>, group<1>>(Stride, Q))
      return 1;
    if (test<bool, group<1>>(Stride, Q))
      return 1;
    if (test<vec<bool, 1>, group<1>>(Stride, Q))
      return 1;
    if (test<vec<bool, 4>, group<1>>(Stride, Q))
      return 1;
    if (test<cl::sycl::cl_bool, group<1>>(Stride, Q))
      return 1;
    if (test<std::byte, group<1>>(Stride, Q))
      return 1;
  }

  if (Q.get_device().get_backend() == backend::host) {
    std::cout << "Test passed.\n";
    std::cout << "Host device: subgroup tests skipped.\n";
    return 0;
  }

  for (int Stride = 1; Stride < WorkGroupSize; Stride++) {

    if (test<int, sub_group>(Stride, Q))
      return 1;
    if (test<int2, sub_group>(Stride, Q))
      return 1;
    if (test<int4, sub_group>(Stride, Q))
      return 1;
    if (test<uint, sub_group>(Stride, Q))
      return 1;
    if (test<uint2, sub_group>(Stride, Q))
      return 1;
    if (test<uint4, sub_group>(Stride, Q))
      return 1;
    if (test<double, sub_group>(Stride, Q))
      return 1;
    if (test<double2, sub_group>(Stride, Q))
      return 1;
    if (test<float, sub_group>(Stride, Q))
      return 1;
    if (test<float2, sub_group>(Stride, Q))
      return 1;
    if (test<float4, sub_group>(Stride, Q))
      return 1;
    if (test<long, sub_group>(Stride, Q))
      return 1;
    if (test<long2, sub_group>(Stride, Q))
      return 1;
    if (test<ulong, sub_group>(Stride, Q))
      return 1;
    if (test<ulong2, sub_group>(Stride, Q))
      return 1;
    if (test<char4, sub_group>(Stride, Q))
      return 1;
    if (test<char8, sub_group>(Stride, Q))
      return 1;
    if (test<char16, sub_group>(Stride, Q))
      return 1;
    if (test<schar4, sub_group>(Stride, Q))
      return 1;
    if (test<schar8, sub_group>(Stride, Q))
      return 1;
    if (test<schar16, sub_group>(Stride, Q))
      return 1;
    if (test<uchar4, sub_group>(Stride, Q))
      return 1;
    if (test<uchar8, sub_group>(Stride, Q))
      return 1;
    if (test<uchar16, sub_group>(Stride, Q))
      return 1;
    if (test<short2, sub_group>(Stride, Q))
      return 1;
    if (test<short4, sub_group>(Stride, Q))
      return 1;
    if (test<short8, sub_group>(Stride, Q))
      return 1;
    if (test<ushort2, sub_group>(Stride, Q))
      return 1;
    if (test<ushort4, sub_group>(Stride, Q))
      return 1;
    if (test<ushort8, sub_group>(Stride, Q))
      return 1;
    if (test<half2, sub_group>(Stride, Q))
      return 1;
    if (test<half4, sub_group>(Stride, Q))
      return 1;
    if (test<half8, sub_group>(Stride, Q))
      return 1;

    if (test<vec<int, 1>, sub_group>(Stride, Q))
      return 1;
    if (test<bool, sub_group>(Stride, Q))
      return 1;
    if (test<vec<bool, 1>, sub_group>(Stride, Q))
      return 1;
    if (test<vec<bool, 4>, sub_group>(Stride, Q))
      return 1;
    if (test<cl::sycl::cl_bool, sub_group>(Stride, Q))
      return 1;
    if (test<std::byte, sub_group>(Stride, Q))
      return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}