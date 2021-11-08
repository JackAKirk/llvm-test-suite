// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.run
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run
// RUN: env SYCL_DEVICE_FILTER=host %t.run
//
// Expected to crash on AMD
// XFAIL: hip_amd
//
// This test is designed to fail if wait_for(sg, E) breaks.

//#include "common.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

template <typename T> class KernelName;

const size_t NElems = 1024;

template <typename T> struct is_vec : std::false_type {};
template <typename T, size_t N> struct is_vec<vec<T, N>> : std::true_type {};

template <typename T> bool checkEqual(vec<T, 1> A, size_t B) {
  T TB = B;
  return A.s0() == TB;
}

template <typename T> bool checkEqual(vec<T, 4> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB && A.w() == TB;
}

template <typename T>
typename std::enable_if_t<!is_vec<T>::value, bool> checkEqual(T A, size_t B) {
  T TB = static_cast<T>(B);
  return A == TB;
}

template <typename T> std::string toString(vec<T, 1> A) {
  std::string R("(");
  return R + std::to_string(A.s0()) + ")";
}

template <typename T> std::string toString(vec<T, 4> A) {
  std::string R("(");
  R += std::to_string(A.x()) + "," + std::to_string(A.y()) + "," +
       std::to_string(A.z()) + "," + std::to_string(A.w()) + ")";
  return R;
}

template <typename T = void>
typename std::enable_if_t<
    !is_vec<T>::value && std::is_same<T, std::byte>::value, std::string>
toString(T A) {
  return std::to_string((unsigned char)A);
}

template <typename T = void>
typename std::enable_if_t<
    !is_vec<T>::value && !std::is_same<T, std::byte>::value, std::string>
toString(T A) {
  return std::to_string(A);
}

template <typename T> int checkResults(buffer<T, 1> &OutBuf) {
  auto Out = OutBuf.template get_access<access::mode::read>();
  int EarlyFailout = 20;

  for (size_t J = 0; J < NElems; J++) {
    size_t ExpectedVal = J * 2;
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

  buffer<T, 1> InBuf(NElems);
  buffer<T, 1> OutBuf(NElems);
  buffer<size_t, 1> SGSizeBuf(1);

  {
    auto Acc = OutBuf.template get_access<access::mode::write>();
    for (size_t I = 0; I < NElems; I++){
      Acc[I] = static_cast<T>(0);
    }
  }
  {
	  auto Acc = InBuf.template get_access<access::mode::write>();
	  for (size_t J = 0; J < NElems; J++){
		Acc[J] = J;
	  }
  }

  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Out = OutBuf.template get_access<access::mode::write>(CGH);
     auto SGSize = SGSizeBuf.template get_access<access::mode::write>(CGH);
     accessor<T, 1, access::mode::read_write, access::target::local> Local(
         range<1>{NElems}, CGH);

     nd_range<1> NDR{range<1>(NElems), range<1>(NElems)};
     CGH.parallel_for<KernelName<T>>(NDR, [=](nd_item<1> NDId) {
       Local[NDId.get_local_id()] = 1;

       auto sg = NDId.get_sub_group();
	   size_t loc = sg.get_local_linear_id();
	   size_t sg_size = sg.get_group_linear_range();
	   size_t gid = sg.get_group_linear_id();
       size_t NElemsToCopy = NElems / 2;
	   auto mask = detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>((1u<<(sg_size/2))-1, sg.get_max_local_range()[0]);

       if(loc<sg_size/2){
         auto E = sycl::ext::oneapi::async_group_copy(
             sg, Local.get_pointer(), In.get_pointer() + gid * sg_size, NElems, mask);
         sycl::ext::oneapi::wait_for(sg, mask, E);

         Local[sg_size - loc] *= 2;
       }

       if(loc<sg_size/2){
         auto E = sycl::ext::oneapi::async_group_copy(
             sg, Out.get_pointer() + gid * sg_size, Local.get_pointer(), NElems, mask);
         sycl::ext::oneapi::wait_for(sg, mask, E);
       }
	   SGSize[0] = sg_size;
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