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

    // note: practically it could also be good to provide clear directions to
    // documentation showing users how to make sure they are constructing queues
    // using distinct devices.

    auto Dev0 = Queues[0].get_device();
    auto Dev1 = Queues[1].get_device();

    assert(Dev0 != Dev1);

    int *arr0 = malloc<int>(N, Queues[0], usm::alloc::device);
    int *arr1 = malloc<int>(N, Queues[1], usm::alloc::device);

    // note: in real use would obviously load/set arr0/arr1 with meaningful
    // data.

    if (Dev0.ext_oneapi_can_access_peer(
            Dev1, sycl::ext::oneapi::peer_access::access_supported)) {
      // dev0 enables itself to access dev1
      Dev0.ext_oneapi_enable_peer_access(Dev1);
      // dev1 enables itself to access dev0
      Dev1.ext_oneapi_enable_peer_access(Dev0);
      // Dev0.ext_oneapi_disable_peer_access(Dev1);
    }
    
    // access Device/Queue 1
    Queues[0].submit([&](handler &cgh) {
      auto myRange = range<1>(N);
      auto myKernel = ([=](id<1> idx) {
        arr0[idx] = idx[0];
        arr1[idx] = idx[0];
      });

      cgh.parallel_for<class p2p_access>(myRange, myKernel);
    }).wait();

    sycl::free(arr0, Queues[0]);
    sycl::free(arr1, Queues[1]);

    if (Dev0.ext_oneapi_can_access_peer(
            Dev1, sycl::ext::oneapi::peer_access::access_supported)) {
      Dev0.ext_oneapi_disable_peer_access(Dev1);
      Dev1.ext_oneapi_disable_peer_access(Dev0);
    }
  }

  //TODO we should also test inter-device atomics
  // Devs[0].ext_oneapi_can_access_peer(Devs[1], sycl::ext::oneapi::peer_access::atomics_supported);

  return 0;
}
