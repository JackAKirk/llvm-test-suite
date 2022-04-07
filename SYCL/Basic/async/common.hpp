#include <CL/sycl.hpp>
#include <iostream>
#include <typeinfo>

using namespace cl::sycl;

template <typename T> class TypeHelper;

template <typename T>
using KernelName = class TypeHelper<typename std::conditional<
    std::is_same<T, std::byte>::value, unsigned char, T>::type>;

// Define the number of work items to enqueue.
const size_t NElems = 32;
const size_t WorkGroupSize = 8;
const size_t NWorkGroups = NElems / WorkGroupSize;

template <typename T> void initInputBuffer(buffer<T, 1> &Buf, size_t Stride) {
  auto Acc = Buf.template get_access<access::mode::write>();
  for (size_t I = 0; I < Buf.size(); I += WorkGroupSize) {
    for (size_t J = 0; J < WorkGroupSize; J++)
      Acc[I + J] = static_cast<T>(I + J + ((J % Stride == 0) ? 100 : 0));
  }
}

template <typename T> void initOutputBuffer(buffer<T, 1> &Buf) {
  auto Acc = Buf.template get_access<access::mode::write>();
  for (size_t I = 0; I < Buf.size(); I++)
    Acc[I] = static_cast<T>(0);
}

template <typename T> struct is_vec : std::false_type {};
template <typename T, size_t N> struct is_vec<vec<T, N>> : std::true_type {};

template <typename T>
typename std::enable_if_t<!is_vec<T>::value, bool> checkEqual(T A, size_t B) {
  T TB = static_cast<T>(B);
  return A == TB;
}

template <typename T> bool checkEqual(vec<T, 1> A, size_t B) {
  T TB = B;
  return A.s0() == TB;
}

template <typename T> bool checkEqual(vec<T, 2> A, size_t B) {
  T TB = B;
  return A.s0() == TB && A.s1() == TB;
}

template <typename T> bool checkEqual(vec<T, 4> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB && A.w() == TB;
}

template <typename T> bool checkEqual(vec<T, 8> A, size_t B) {
  T TB = B;
  return A.s0() == TB && A.s1() == TB && A.s2() == TB && A.s3() == TB &&
         A.s4() == TB && A.s5() == TB && A.s6() == TB && A.s7() == TB;
}

template <typename T> bool checkEqual(vec<T, 16> A, size_t B) {
  T TB = B;
  return A.s0() == TB && A.s1() == TB && A.s2() == TB && A.s3() == TB &&
         A.s4() == TB && A.s5() == TB && A.s6() == TB && A.s7() == TB &&
         A.s8() == TB && A.s9() == TB && A.sA() == TB && A.sB() == TB &&
         A.sC() == TB && A.sD() == TB && A.sE() == TB && A.sF() == TB;
}

template <typename T> std::string toString(vec<T, 1> A) {
  std::string R("(");
  return R + std::to_string(A.s0()) + ")";
}

template <typename T> std::string toString(vec<T, 2> A) {
  std::string R("(");
  return R + std::to_string(A.s0()) + "," + std::to_string(A.s1()) + ")";
}

template <typename T> std::string toString(vec<T, 4> A) {
  std::string R("(");
  R += std::to_string(A.x()) + "," + std::to_string(A.y()) + "," +
       std::to_string(A.z()) + "," + std::to_string(A.w()) + ")";
  return R;
}

template <typename T> std::string toString(vec<T, 8> A) {
  std::string R("(");
  return R + std::to_string(A.s0()) + "," + std::to_string(A.s1()) + "," +
         std::to_string(A.s2()) + "," + std::to_string(A.s3()) + "," +
         std::to_string(A.s4()) + "," + std::to_string(A.s5()) + "," +
         std::to_string(A.s6()) + "," + std::to_string(A.s7()) + ")";
}

template <typename T> std::string toString(vec<T, 16> A) {
  std::string R("(");
  return R + std::to_string(A.s0()) + "," + std::to_string(A.s1()) + "," +
         std::to_string(A.s2()) + "," + std::to_string(A.s3()) + "," +
         std::to_string(A.s4()) + "," + std::to_string(A.s5()) + "," +
         std::to_string(A.s6()) + "," + std::to_string(A.s7()) + "," +
         std::to_string(A.s8()) + "," + std::to_string(A.s9()) + "," +
         std::to_string(A.sA()) + "," + std::to_string(A.sB()) + "," +
         std::to_string(A.sC()) + "," + std::to_string(A.sD()) + "," +
         std::to_string(A.sE()) + "," + std::to_string(A.sF()) + ")";
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

template <typename T> int checkResults(buffer<T, 1> &OutBuf, size_t Stride) {
  auto Out = OutBuf.template get_access<access::mode::read>();
  int EarlyFailout = 20;

  for (size_t I = 0; I < OutBuf.size(); I += WorkGroupSize) {
    for (size_t J = 0; J < WorkGroupSize; J++) {
      size_t ExpectedVal = (J % Stride == 0) ? (100 + I + J) : 0;
      if (!checkEqual(Out[I + J], ExpectedVal)) {
        std::cerr << std::string(typeid(T).name()) + ": Stride=" << Stride
                  << " : Incorrect value at index " << I + J
                  << " : Expected: " << toString(ExpectedVal)
                  << ", Computed: " << toString(Out[I + J]) << "\n";
        if (--EarlyFailout == 0)
          return 1;
      }
    }
  }
  return EarlyFailout - 20;
}

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
