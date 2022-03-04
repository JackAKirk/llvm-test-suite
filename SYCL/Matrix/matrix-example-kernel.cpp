#include <CL/sycl.hpp>
#include <math.h>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// This example performs a matrix Multiplication and Add (MAD) operation
// followed by element wise operations on the output "D" matrix. A common class
// of element wise operations used in neural networks are activation functions.
// In this example the tanh (hyperbolic tangent) activation function is used on
// every element of the output matrix register "fragments" directly after the
// MAD operation. The "A", "B", and "accumulator" matrices ("Big Matrices") are
// stored in global memory. Each accumulator matrix consists of multiple
// "TILEs".  Each TILE is handled by a single work-group; in total there are
// (TILES_M * TILES_N) TILES per "accumulator" matrix.  Each "accumulator" TILE
// consists of (Sub_Tiles_M * Sub_Tiles_N) sub-tiles. Each output "accumulator"
// ("D") TILE is calculated by a MAD operation on an "A" TILE consisting of
// (Sub_Tiles_M * Sub_Tiles_K) sub-tiles with a "B" TILE consisting of
// (Sub_Tiles_K * Sub_Tiles_N) sub-tiles, accumulated with the corresponding
// "accumulator" TILE. In this example each sub-group is responsible for
// calculating the result of a single sub-tile of the "accumulator matrix". In
// this example each "accumulator" TILE is first loaded from the global "Big
// Matrix" array to shared memory; each work-item copies two elements of each
// TILE to shared memory. "A"/"B" TILES are only represented in global memory in
// this example; however further optimization can be achieved by also
// representing these matrices in shared memory within the kernel. As a further
// performance optimization one can add shared memory paddings to avoid bank
// conflicts: see e.g. Faingneart et al, https://arxiv.org/pdf/2009.12263v4.pdf.
// In general the most optimal kernel setup will depend on the sizes and types
// of the matrices involved. This example uses row_major a, b, and accumulator
// matrices. Double type is used for all matrices.

// M, N, K define the unit sizes of dimensions of the three types (a, b,
// accumulator) of matrices per subgroup operation:
// M: number of rows of "C"/"D" (Accumulator) sub-matrices,
// number of cols of "B" sub-matrix.
// N: number of cols of "C"/"D" (Accumulator) sub-matrices,
// number of rows of "A" sub-matrix.
// K: number of cols of "A"/number of rows of "B" sub-matrices.

constexpr int N_WORKITEMS_PER_MATRIX_OP =
    32; // the number of work-items per MAD subgroup is always 32 for Nvidia.

// The number of sub-tiles computed per work-group should be optimized per
// architecture taking into account shared memory usage.
constexpr int Sub_Tiles_M = 4;  // number of submatrices per row of accumulator
                                // ("C", "D") TILE matrices.
constexpr int Sub_Tiles_N = 4;  // number of submatrices per col of accumulator
                                // TILE matrices.
constexpr int Sub_Tiles_K = 16; // number of submatrices per col of "A"/per row
                                // of "B", TILES.

constexpr int TILES_M = 20; // number of TILES in the accumulator
                            // ("C", "D") "Big Matrix" in the M dimension.
constexpr int TILES_N = 20; // number of "TILES in the accumulator
                            // ("C", "D") "Big Matrix" in the N dimension.

template <typename T1, typename T2, size_t M, size_t K, size_t N>
class TypeHelper;

template <typename T1, typename T2, size_t M, size_t K, size_t N>
using KernelName = class TypeHelper<T1, T2, M, K, N>;

// This function performs a reference MAD + tanh calculation on the host
template <typename T1, typename T2, size_t Big_N, size_t Big_K>
T2 matrix_ref_mn(const int &m, const int &n, T1 *A, T1 *B, T2 *C) {
  T2 res = C[m * Big_N + n];

  if constexpr (std::is_same<T1, uint16_t>::value) {
    for (int k = 0; k < Big_K; k++)
      res += make_fp32(A[m * Big_K + k]) * make_fp32(B[k * Big_N + n]);
  } else {
    for (int k = 0; k < Big_K; k++)

      res +=
          static_cast<T2>(A[m * Big_K + k]) * static_cast<T2>(B[k * Big_N + n]);
  }

  return tanh(res);
}

template <typename T1, typename T2, size_t M, size_t K, size_t N> void test() {

  constexpr auto Big_M =
      TILES_M * Sub_Tiles_M * M; // total number of M dimension matrix elements
                                 // for the A/B/Accumulator "Big matrices".
  constexpr auto Big_N =
      TILES_N * Sub_Tiles_N * N; // total number of N dimension matrix elements
                                 // for the A/B/Accumulator "Big matrices".
  constexpr auto Big_K =
      Sub_Tiles_K * K; // total number of K dimension matrix elements for the
                       // A/B/Accumulator "Big matrices".

  T1 A[Big_M * Big_K];
  T1 B[Big_K * Big_N];
  T2 C[Big_M * Big_N];
  T2 D[Big_M * Big_N];

  for (int i = 0; i < Big_M * Big_N; i++) {
    C[i] = 1;
    D[i] = 0;
  }

  for (int i = 0; i < Big_M * Big_K; i++) {
    A[i] = i;
  }

  for (int i = 0; i < Big_K * Big_N; i++) {
    B[i] = i;
  }

  buffer<T1, 1> bufA(A, range<1>(Big_M * Big_K));
  buffer<T1, 1> bufB(B, range<1>(Big_K * Big_N));
  buffer<T2, 1> bufC(C, range<1>(Big_M * Big_N));
  buffer<T2, 1> bufD(D, range<1>(Big_M * Big_N));

  queue q;
  q.submit([&](handler &cgh) {
    auto accC = bufC.template get_access<access::mode::read_write>(cgh);
    auto accA = bufA.template get_access<access::mode::read_write>(cgh);
    auto accB = bufB.template get_access<access::mode::read_write>(cgh);
    auto accD = bufD.template get_access<access::mode::read_write>(cgh);
    accessor<T2, 1, access::mode::read_write, access::target::local> LocalC(
        range<1>{Sub_Tiles_M * M * Sub_Tiles_N * N}, cgh);

    range<2> LocalRange = {Sub_Tiles_M,
                           Sub_Tiles_N * N_WORKITEMS_PER_MATRIX_OP};
    range<2> GlobalRange = {TILES_M * Sub_Tiles_M,
                            TILES_N * Sub_Tiles_N * N_WORKITEMS_PER_MATRIX_OP};

    cgh.parallel_for<KernelName<T1, T2, M, K, N>>(
        nd_range<2>(GlobalRange, LocalRange), [=](nd_item<2> item) {
          auto Group = item.get_group();
          auto group_id_m = Group.get_group_id()[0];
          auto group_id_n = Group.get_group_id()[1];
          sycl::sub_group sg = item.get_sub_group();

          const auto m = item.get_local_id()[0];
          const auto n = item.get_local_id()[1] / 32;

          //copy to shared memory
#pragma unroll
          for (int m_c = 0; m_c < 2; m_c++) {
            LocalC[(m * M + (m_c * Sub_Tiles_N + n)) * Sub_Tiles_N * N] =
                accC[((group_id_m * Sub_Tiles_M * M) +
                      (m * M + (m_c * Sub_Tiles_N + n))) *
                         Big_N +
                     group_id_n * Sub_Tiles_N * N];
          }

          joint_matrix<T1, matrix_use::a, M, K, matrix_layout::row_major> sub_a;

          joint_matrix<T1, matrix_use::b, K, N, matrix_layout::row_major> sub_b;

          joint_matrix<T2, matrix_use::accumulator, M, N,
                       matrix_layout::row_major>
              sub_c;

          // using shared memory route
          joint_matrix_load(
              sg, sub_c, LocalC.get_pointer() + (m * M * Sub_Tiles_N * N) + (n * N),
              Sub_Tiles_N * N);

          for (int k = 0; k < Sub_Tiles_K;
               k++) // row/col id of current submatrix of A/B TILE matrices
          {
            joint_matrix_load(sg, sub_a,
                              accA.get_pointer() + (k * K) +
                                  ((group_id_m * Sub_Tiles_M) + m) * M * Big_K,
                              Big_K);

            joint_matrix_load(sg, sub_b,
                              accB.get_pointer() + (k * K * Big_N) +
                                  (((group_id_n * Sub_Tiles_N) + n) * N),
                              Big_N);

            sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          }
          // Element wise operations
          // Common deep learning use cases act on every element in the matrix.
          // We act directly on joint_matrix register fragments. In the case of
          // double each fragment consists of two elements. Each work-item holds
          // a fragment.
          for (int i = 0; i < 2; i++) {
            sub_c.data[i] = sycl::tanh(sub_c.data[i]);
          }

          joint_matrix_store(sg, sub_c,
                             accD.get_pointer() +
                                 (((group_id_m * Sub_Tiles_M) + m) * M) *
                                     Big_N +
                                 ((group_id_n * Sub_Tiles_N) + n) * N,
                             Big_N);
        });
  });

  q.wait();

  const auto host_accessor = bufD.template get_access<access::mode::read>();
  for (int m = 0; m < Big_M; m++)
    for (int n = 0; n < Big_N; n++) {
      assert((host_accessor[m * Big_N + n] ==
              matrix_ref_mn<T1, T2, Big_N, Big_K>(m, n, A, B, C)));
    }
};

int main() {

  test<double, double, 8, 4, 8>();

  return 0;
};