#include <CL/sycl.hpp>

std::vector<sycl::queue> getQueues() {

  sycl::platform nv;

  for (const auto &plt : sycl::platform::get_platforms()) {

    if (plt.get_backend() == cl::sycl::backend::cuda)
      nv = plt;
    break;
  }
  std::vector<sycl::queue> Queues;
  auto Devices = nv.get_devices();
  if (Devices.size() > 1) {
    std::transform(Devices.begin(), Devices.end(), std::back_inserter(Queues),
                   [](const sycl::device &D) { return sycl::queue{D}; });
  } else {
    Queues.push_back(sycl::queue{});
    Queues.push_back(sycl::queue{});
  }
  assert(Queues[0].get_context() != Queues[1].get_context());

  return Queues;
}

void remind(const long &width, const long &height, const long &depth) {
  /*
    https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBufferRect.html

    buffer_origin defines the (x, y, z) offset in the memory region associated
    with buffer. For a 2D rectangle region, the z value given by
    buffer_origin[2] should be 0. The offset in bytes is computed as
    buffer_origin[2] × buffer_slice_pitch + buffer_origin[1] × buffer_row_pitch
    + buffer_origin[0].

    region defines the (width in bytes, height in rows, depth in slices) of the
    2D or 3D rectangle being read or written. For a 2D rectangle copy, the depth
    value given by region[2] should be 1. The values in region cannot be 0.


    buffer_row_pitch is the length of each row in bytes to be used for the
    memory region associated with buffer. If buffer_row_pitch is 0,
    buffer_row_pitch is computed as region[0].

    buffer_slice_pitch is the length of each 2D slice in bytes to be used for
    the memory region associated with buffer. If buffer_slice_pitch is 0,
    buffer_slice_pitch is computed as region[1] × buffer_row_pitch.
  */
  std::cout << "For BUFFERS" << std::endl;
  std::cout << "         Region SHOULD be : " << width * sizeof(float) << "/"
            << height << "/" << depth << std::endl; // 64/5/3
  std::cout << "  RowPitch SHOULD be 0 or : " << width * sizeof(float)
            << std::endl; // 0 or 64
  std::cout << "SlicePitch SHOULD be 0 or : " << width * sizeof(float) * height
            << std::endl
            << std::endl; // 0 or 320
}

void remindImage(const long &width, const long &height, const long &depth) {
  /*
    https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadImage.html

    row_pitch in clEnqueueReadImage and input_row_pitch in clEnqueueWriteImage
    is the length of each row in bytes. This value must be greater than or equal
    to the element size in bytes × width. If row_pitch (or input_row_pitch) is
    set to 0, the appropriate row pitch is calculated based on the size of each
    element in bytes multiplied by width.

    slice_pitch in clEnqueueReadImage and input_slice_pitch in
    clEnqueueWriteImage is the size in bytes of the 2D slice of the 3D region of
    a 3D image or each image of a 1D or 2D image array being read or written
    respectively.
  */

  std::cout << "For IMAGES" << std::endl;
  std::cout << "           Region SHOULD be : " << width << "/" << height << "/"
            << depth << std::endl; // 16/5/3
  std::cout << "   row_pitch SHOULD be 0 or : " << width * sizeof(sycl::float4)
            << std::endl; // 0 or 256
  std::cout << " slice_pitch SHOULD be 0 or : "
            << width * sizeof(sycl::float4) * height << std::endl
            << std::endl; // 0 or 1280
}
