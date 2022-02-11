// RUN: %clangxx -fsycl -std=c++17 -fsycl-targets=%sycl_triple %s -o %t.run
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run
// RUN: env SYCL_DEVICE_FILTER=host %t.run

#include "common.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

template <typename T> class TypeHelper;

template <typename T>
using KernelName = class TypeHelper<typename std::conditional<
    std::is_same<T, std::byte>::value, unsigned char, T>::type>;

template <typename T> int test(size_t Stride) {
  queue Q;

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
     CGH.parallel_for<KernelName<T>>(NDR, [=](nd_item<1> NDId) {
       auto GrId = NDId.get_group_linear_id();
       auto Group = NDId.get_group();
       size_t NElemsToCopy =
           WorkGroupSize / Stride + ((WorkGroupSize % Stride) ? 1 : 0);
       size_t Offset = GrId * WorkGroupSize;
       if (Stride == 1) { // Check the version without stride arg.
         auto E = NDId.async_work_group_copy(
             Local.get_pointer(), In.get_pointer() + Offset, NElemsToCopy);
         E.wait();
       } else {
         auto E = NDId.async_work_group_copy(Local.get_pointer(),
                                             In.get_pointer() + Offset,
                                             NElemsToCopy, Stride);
         E.wait();
       }

       if (Stride == 1) { // Check the version without stride arg.
         auto E = Group.async_work_group_copy(
             Out.get_pointer() + Offset, Local.get_pointer(), NElemsToCopy);
         Group.wait_for(E);
       } else {
         auto E = Group.async_work_group_copy(Out.get_pointer() + Offset,
                                              Local.get_pointer(), NElemsToCopy,
                                              Stride);
         Group.wait_for(E);
       }
     });
   }).wait();

  return checkResults(OutBuf, Stride);
}

int main() {
  for (int Stride = 1; Stride < WorkGroupSize; Stride++) {
    if (test<int>(Stride))
      return 1;
    if (test<int2>(Stride))
      return 1;
    if (test<int4>(Stride))
      return 1;
    if (test<uint>(Stride))
      return 1;
    if (test<uint2>(Stride))
      return 1;
    if (test<uint4>(Stride))
      return 1;
    if (test<double>(Stride))
      return 1;
    if (test<double2>(Stride))
      return 1;
    if (test<float>(Stride))
      return 1;
    if (test<float2>(Stride))
      return 1;
    if (test<float4>(Stride))
      return 1;
    if (test<long>(Stride))
      return 1;
    if (test<long2>(Stride))
      return 1;
    if (test<ulong>(Stride))
      return 1;
    if (test<ulong2>(Stride))
      return 1;
    if (test<char4>(Stride))
      return 1;
    if (test<char8>(Stride))
      return 1;
    if (test<char16>(Stride))
      return 1;
    if (test<schar4>(Stride))
      return 1;
    if (test<schar8>(Stride))
      return 1;
    if (test<schar16>(Stride))
      return 1;
    if (test<uchar4>(Stride))
      return 1;
    if (test<uchar8>(Stride))
      return 1;
    if (test<uchar16>(Stride))
      return 1;
    if (test<short2>(Stride))
      return 1;
    if (test<short4>(Stride))
      return 1;
    if (test<short8>(Stride))
      return 1;
    if (test<ushort2>(Stride))
      return 1;
    if (test<ushort4>(Stride))
      return 1;
    if (test<ushort8>(Stride))
      return 1;
    if (test<half2>(Stride))
      return 1;
    if (test<half4>(Stride))
      return 1;
    if (test<half8>(Stride))
      return 1;

    if (test<vec<int, 1>>(Stride))
      return 1;
    if (test<int4>(Stride))
      return 1;
    if (test<bool>(Stride))
      return 1;
    if (test<vec<bool, 1>>(Stride))
      return 1;
    if (test<vec<bool, 4>>(Stride))
      return 1;
    if (test<cl::sycl::cl_bool>(Stride))
      return 1;
    if (test<std::byte>(Stride))
      return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}
