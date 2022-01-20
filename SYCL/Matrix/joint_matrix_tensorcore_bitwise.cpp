// REQUIRES: gpu, cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_75 -DSYCL_EXT_ONEAPI_MATRIX=3  %s -o %t.out
//
// Specifying the sm version via the --cuda-gpu-arch flag is necessary
// for the Nvidia case.  DPC++ JIT compilation is not
// supported for the Nvidia matrix extension, although some JIT optimizations
// are performed at the level of the PTX assembly code.

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// Example usage of Nvidia matrix multiply.
// Optimizations such as memory paddings for avoiding bank conflicts are not
// included in this test which aids clarity for what is going on. This example
// forms a "Big matrix" corresponding to a single "TILE" using cuda example
// terminology.  Multiple TILES can be used to construct yet larger matrices.
// This example uses row_major a, b, and accumulator matrices.

// M, N, K define the unit sizes of dimensions of the three types (a, b,
// accumulator) of matrices per subgroup operation:
// M: number of rows of "C"/"D" (Accumulator) sub-matrices,
// number of cols of "B" sub-matrix.
// N: number of cols of "C"/"D" (Accumulator) sub-matrices,
// number of rows of "A" sub-matrix.
// K: number of cols of "A"/number of rows of "B" sub-matrices.

constexpr int N_THREADS_PER_MATRIX_OP =
    32; // the number of threads per MMA subgroup is always 32 for Nvidia.

constexpr int SUB_TILES_M =
    2; // number of submatrices per row of accumulator ("C", "D") matrices.
constexpr int SUB_TILES_N =
    3; // number of submatrices per col of accumulator matrices.
constexpr int SUB_TILES_K =
    4; // number of submatrices per col of "A"/per row of "B", matrices.

template <size_t M, size_t K, size_t N, class BinaryOperation> class TypeHelper;

template <size_t M, size_t K, size_t N, class BinaryOperation>
using KernelName = class TypeHelper<M, K, N, BinaryOperation>;

template <size_t Big_N, size_t Big_K, class BinaryOperation>
int32_t matrix_ref_mn(const int &m, const int &n, uint32_t *A, uint32_t *B,
                      int32_t *C, BinaryOperation Op) {
  int32_t res = C[m * Big_N + n];

  {
    for (int k = 0; k < Big_K / 32; k++)
      if constexpr (std::is_same<BinaryOperation,
                                 sycl::bit_and<uint32_t>>::value) {
        res += popcount(A[m * Big_K / 32 + k] & B[n * Big_K / 32 + k]);
      } else if constexpr (std::is_same<BinaryOperation,
                                        sycl::bit_xor<uint32_t>>::value) {
        res += popcount(A[m * Big_K / 32 + k] ^ B[n * Big_K / 32 + k]);
      } else {
        throw std::runtime_error(
            "Only sycl::bit_xor<uint32_t> and sycl::bit_and<uint32_t> "
            "operators are currently supported for binary matrix "
            "multiplication and addition.");
      }
  }
  return res;
}

template <size_t Sub_Tiles_M, size_t Sub_Tiles_K, size_t Sub_Tiles_N, size_t M,
          size_t K, size_t N, class BinaryOperation>
void test(BinaryOperation Op) {

  constexpr auto Big_M =
      Sub_Tiles_M *
      M; // total number of M dimension matrix elements for the "Big matrix".
  constexpr auto Big_N =
      Sub_Tiles_N *
      N; // total number of N dimension matrix elements for the "Big matrix".
  constexpr auto Big_K =
      Sub_Tiles_K *
      K; // total number of K dimension matrix elements for the "Big matrix".

  uint32_t A[Big_M * Big_K / 32];
  uint32_t B[Big_K * Big_N / 32];
  int32_t C[Big_M * Big_N];
  int32_t D[Big_M * Big_N];

  for (int i = 0; i < Big_M * Big_N; i++) {
    C[i] = 1;
    D[i] = 0;
  }

  srand(time(NULL));

  for (int i = 0; i < Big_M * Big_K / 32; i++) {
    A[i] = (uint32_t)rand();
  }

  for (int i = 0; i < Big_K * Big_N / 32; i++) {
    B[i] = (uint32_t)rand();
  }

  buffer<uint32_t, 1> bufA(A, range<1>(Big_M * Big_K / 32));
  buffer<uint32_t, 1> bufB(B, range<1>(Big_K * Big_N / 32));
  buffer<int32_t, 1> bufC(C, range<1>(Big_M * Big_N));
  buffer<int32_t, 1> bufD(D, range<1>(Big_M * Big_N));

  queue q;
  q.submit([&](handler &cgh) {
    auto accC = bufC.template get_access<access::mode::read_write>(cgh);
    auto accA = bufA.template get_access<access::mode::read_write>(cgh);
    auto accB = bufB.template get_access<access::mode::read_write>(cgh);
    auto accD = bufD.template get_access<access::mode::read_write>(cgh);

    range<2> LocalRange = {1, N_THREADS_PER_MATRIX_OP};
    range<2> GlobalRange = {Sub_Tiles_M, Sub_Tiles_N * N_THREADS_PER_MATRIX_OP};

    cgh.parallel_for<KernelName<
        M, K, N, BinaryOperation>>(nd_range<2>(GlobalRange, LocalRange), [=
    ](nd_item<2> item)[[sycl::reqd_work_group_size(1, 1, 32)]] {
      sycl::sub_group sg = item.get_sub_group();
      const auto m =
          item.get_group()
              .get_id()[0]; // row id of current submatrix of BIG C matrix
      const auto n = item.get_group().get_id()[1]; // column id of current
                                                   // submatrix of BIG C matrix
      // matrix_use::a must have matrix_layout::row_major for single-bit cases
      joint_matrix<uint32_t, matrix_use::a, 8, 128, matrix_layout::row_major>
          sub_a;
      // matrix_use::b must have matrix_layout::col_major for single-bit cases
      joint_matrix<uint32_t, matrix_use::b, 128, 8, matrix_layout::col_major>
          sub_b;

      joint_matrix<int32_t, matrix_use::accumulator, 8, 8,
                   matrix_layout::row_major>
          sub_c;

      joint_matrix_load(sg, sub_c, accC.get_pointer() + (m * M) * Big_N + n * N,
                        Big_N);

      for (int k = 0; k < Sub_Tiles_K;
           k++) // row/col id of current submatrix of BIG A/B matrices
      {
        joint_matrix_load(
            sg, sub_a, accA.get_pointer() + (k * K / 32) + (m * M * Big_K / 32),
            Big_K);

        joint_matrix_load(
            sg, sub_b, accB.get_pointer() + (n * N * Big_K / 32) + (k * K / 32),
            Big_K);

        sub_c = joint_matrix_bmad(sg, sub_a, sub_b, sub_c, Op);
      }
      joint_matrix_store(sg, sub_c,
                         accD.get_pointer() + (m * M) * Big_N + n * N, Big_N);
    });
  });

  q.wait();

  const auto host_accessor = bufD.template get_access<access::mode::read>();
  for (int m = 0; m < Big_M; m++)
    for (int n = 0; n < Big_N; n++) {
      assert((host_accessor[m * Big_N + n] ==
              matrix_ref_mn<Big_N, Big_K>(m, n, A, B, C, Op)));
    }
};

int main() {

  test<SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 128, 8>(
      sycl::bit_and<uint32_t>());
  test<SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 128, 8>(
      sycl::bit_xor<uint32_t>());

  return 0;
};
