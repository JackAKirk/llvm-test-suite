// Tests that inter device copy (P2P) works correctly when the copy is made
// across distinct contexts for images.  When a pair of command groups are
// submitted to different queues with distinct contexts the runtime copies image
// memory across contexts. For the cuda backend this copy is made directly
// across devices.

#include "utils.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>
#include <iostream>

using namespace cl::sycl;

constexpr long width = 16;

constexpr long height = 5;
constexpr long total = width * height;

constexpr long depth = 3;
constexpr long total3D = total * depth;

const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

constexpr auto SYCLRead = sycl::access::mode::read;
constexpr auto SYCLWrite = sycl::access::mode::write;

void copyP2P_1D() {

  const sycl::range<1> ImgSize_1D(width);
  std::vector<sycl::float4> data_1D(ImgSize_1D.size(), {1, 2, 3, 4});

  {
    std::cout << "-- 1D" << std::endl;
    sycl::image<1> image_1D(data_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    auto Queues = getQueues();

    Queues[0].submit([&](sycl::handler &CGH) {
      auto readAcc = image_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_1D>(
          ImgSize_1D, [=](sycl::item<1> Item) {
            sycl::float4 Data = readAcc.read(int(Item[0]));
            writeAcc.write(int(Item[0]), Data * 2);
          });
    });

    Queues[1].submit([&](sycl::handler &CGH) {
      auto readAcc = image_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_1D_2nd>(
          ImgSize_1D, [=](sycl::item<1> Item) {
            sycl::float4 Data = readAcc.read(int(Item[0]));
            writeAcc.write(int(Item[0]), Data * 2);
          });
    });
    const auto host_accessor =
        image_1D.get_access<sycl::float4, cl::sycl::access::mode::read>();

    for (int i = 0; i < width; i++) {
      assert(host_accessor.read(i)[0] == 4);
      assert(host_accessor.read(i)[1] == 8);
      assert(host_accessor.read(i)[2] == 12);
      assert(host_accessor.read(i)[3] == 16);
    }
    std::cout << "about to destruct 1D" << std::endl;
  }
}

void copyP2P_2D() {

  const sycl::range<2> ImgSize_2D(width, height);
  std::vector<sycl::float4> data_2D(ImgSize_2D.size(), {7, 7, 7, 7});

  {
    std::cout << "-- 2D" << std::endl;
    sycl::image<2> image_2D(data_2D.data(), ChanOrder, ChanType, ImgSize_2D);
    auto Queues = getQueues();

    Queues[0].submit([&](sycl::handler &CGH) {
      auto readAcc = image_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_2D>(
          ImgSize_2D, [=](sycl::item<2> Item) {
            sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
            writeAcc.write(sycl::int2{Item[0], Item[1]}, Data * 2);
          });
    });

    Queues[1].submit([&](sycl::handler &CGH) {
      auto readAcc = image_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_2D_2nd>(
          ImgSize_2D, [=](sycl::item<2> Item) {
            sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
            writeAcc.write(sycl::int2{Item[0], Item[1]}, Data * 2);
          });
    });
    const auto host_accessor =
        image_2D.get_access<sycl::float4, cl::sycl::access::mode::read>();

    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++) {
        assert(host_accessor.read(sycl::int2{j, i})[0] == 28);
        assert(host_accessor.read(sycl::int2{j, i})[1] == 28);
        assert(host_accessor.read(sycl::int2{j, i})[2] == 28);
        assert(host_accessor.read(sycl::int2{j, i})[3] == 28);
      }
    std::cout << "about to destruct 2D" << std::endl;
  }
}

void copyP2P_3D() {

  const sycl::range<3> ImgSize_3D(width, height, depth);
  std::vector<sycl::float4> data_3D(ImgSize_3D.size(), {11, 11, 11, 11});

  {
    std::cout << "-- 3D" << std::endl;
    sycl::image<3> image_3D(data_3D.data(), ChanOrder, ChanType, ImgSize_3D);
    auto Queues = getQueues();

    Queues[0].submit([&](sycl::handler &CGH) {
      auto readAcc = image_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_3D>(
          ImgSize_3D, [=](sycl::item<3> Item) {
            sycl::float4 Data =
                readAcc.read(sycl::int4{Item[0], Item[1], Item[2], 0});
            writeAcc.write(sycl::int4{Item[0], Item[1], Item[2], 0}, Data * 2);
          });
    });

    Queues[1].submit([&](sycl::handler &CGH) {
      auto readAcc = image_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_3D_2nd>(
          ImgSize_3D, [=](sycl::item<3> Item) {
            sycl::float4 Data =
                readAcc.read(sycl::int4{Item[0], Item[1], Item[2], 0});
            writeAcc.write(sycl::int4{Item[0], Item[1], Item[2], 0}, Data * 2);
          });
    });
    const auto host_accessor =
        image_3D.get_access<sycl::float4, cl::sycl::access::mode::read>();
    for (int i = 0; i < depth; i++)
      for (int j = 0; j < height; j++)
        for (int k = 0; k < width; k++) {
          assert(host_accessor.read(sycl::int4{k, j, i, 0})[0] == 44);
          assert(host_accessor.read(sycl::int4{k, j, i, 0})[1] == 44);
          assert(host_accessor.read(sycl::int4{k, j, i, 0})[2] == 44);
          assert(host_accessor.read(sycl::int4{k, j, i, 0})[3] == 44);
        }
    std::cout << "about to destruct 3D" << std::endl;
  }
}

int main() {
  remindImage(width, height, depth);
  for (int i = 0; i < 5; i++) {
    copyP2P_1D();
    copyP2P_2D();
    copyP2P_3D();
  }
}

//CHECK: -- 1D
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 1 / 1  --  arrSz/row/slice : 0 / 256 / 256  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4340
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 1 / 1  --  arrSz/row/slice : 0 / 0 / 0  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4340
//CHECK: ---> piextEnqueueMemImageCopyPeer(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 1 / 1  --  arrSz/row/slice : 0 / 0 / 0  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4340
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: about to destruct 1D
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: -- 2D
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 5 / 1  --  arrSz/row/slice : 0 / 256 / 1280  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4337
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 5 / 1  --  arrSz/row/slice : 0 / 0 / 0  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4337
//CHECK: ---> piextEnqueueMemImageCopyPeer(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
// CHECK-NEXT: <unknown> : 256
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 5 / 1  --  arrSz/row/slice : 0 / 0 / 0  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4337
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
// CHECK-NEXT: <unknown> : 256
//CHECK: about to destruct 2D
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: -- 3D
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 5 / 3  --  arrSz/row/slice : 0 / 256 / 1280  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4338
//CHECK: ---> piMemImageCreate(
//CHECK: image_desc w/h/d : 16 / 5 / 3  --  arrSz/row/slice : 0 / 0 / 0  --  num_mip_lvls/num_smpls/image_type : 0 / 0 / 4338
//CHECK: ---> piextEnqueueMemImageCopyPeer(
//CHECK: pi_image_region width/height/depth : 16/5/3
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/3
// CHECK-NEXT: <unknown> : 256
// CHECK-NEXT: <unknown> : 1280
//CHECK: about to destruct 3D
