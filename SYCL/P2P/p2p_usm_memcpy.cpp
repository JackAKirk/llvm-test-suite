#include <cassert>
#include <memory>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  int Data[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  {

    std::vector<sycl::device> Devs;
    
    // Note that this code is temporary due to the temporary lack of multiple devices per sycl context in the nvidia backend.
    ////////////////////////
    for (const auto &plt : sycl::platform::get_platforms()) {

      if (plt.get_backend() == sycl::backend::cuda)
        Devs.push_back(plt.get_devices()[0]);
    }
    ////////////////////////
    
    ///// Enable bi-directional peer copies
    Devs[0].ext_oneapi_enable_peer_access(Devs[1]);

    std::vector<sycl::queue> Queues;
    std::transform(Devs.begin(), Devs.end(), std::back_inserter(Queues),
                   [](const sycl::device &D) { return sycl::queue{D}; });

    assert(Queues.size() > 1);

    int N = 100;
    int val = 5;
    int *input = (int *)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) {
      input[i] = val;
    }

    int *arr0 = malloc<int>(N, Queues[0], usm::alloc::device);
    Queues[0].memcpy(arr0, input, N * sizeof(int));
    Queues[0].wait();

    int *arr2 = malloc<int>(N, Queues[1], usm::alloc::device);
    // Copy device usm allocated in devices/cuContexts
    Queues[0].copy(arr1, arr0, N).wait();
    int *out;
    out = new int[N];
    Queues[0].memcpy(out, arr1, N * sizeof(int)).wait();

    sycl::free(arr0, Queues[0]);
    sycl::free(arr1, Queues[1]);

    std::cout << out[0];
    delete[] out;
  }

  return 0;
}
