#include <CL/sycl.hpp>
#include <iostream>
#include <typeinfo>

using namespace cl::sycl;

template <typename T> void initOutputBuffer(buffer<T, 1> &Buf) {
  auto Acc = Buf.template get_access<access::mode::write>();
  for (size_t I = 0; I < Buf.size(); I++)
    Acc[I] = static_cast<T>(0);
}

template <typename T> struct is_vec : std::false_type {};
template <typename T, size_t N> struct is_vec<vec<T, N>> : std::true_type {};

template <typename T> bool checkEqual(vec<T, 1> A, size_t B) {
  T TB = B;
  return A.s0() == TB;
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

template <typename T> bool checkEqual(vec<T, 4> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB && A.w() == TB;
}

template <typename T> bool checkEqual(vec<T, 2> A, size_t B) {
  T TB = B;
  return A.s0() == TB && A.s1() == TB;
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
