#include <iostream>
#include <cuda_profiler_api.h>
#include "device.hpp"
#include "forall.hpp"

#define HOST_DEVICE __host__ __device__

int main(int argc, char *argv[])
{
  int N = 30;
  float kernel_time = 0; // time the kernel should run in ms
#if defined(GPU)
  kernel_time = 20;
  if (argc > 1) kernel_time = atoi(argv[1]);
#endif
  int cuda_device = 0;
  
  cudaDeviceProp deviceProp;
  cudaGetDevice(&cuda_device);
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  if ((deviceProp.concurrentKernels == 0))
  {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }
  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
   deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

#if defined(__arm__) || defined(__aarch64__)
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 1000));
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif


  // -----------------------------------------------------------------------
  //                             Device Stuff
  // -----------------------------------------------------------------------

#if defined(GPU)
  std::cout << "------ Running on GPU ------" << std::endl;
  camp::devices::Context dev1{camp::devices::Cuda()};
  camp::devices::Context dev2{camp::devices::Cuda()};
#else
  std::cout << "------ Running on CPU ------" << std::endl;
  camp::devices::Context dev1{camp::devices::Host()};
  camp::devices::Context dev2{camp::devices::Host()};
#endif

  float * m1 = dev1.allocate<float>(N);
  float * m2 = dev2.allocate<float>(N);

  auto clock_lambda_1 = [=] HOST_DEVICE (int idx) {
    m1[idx] = idx * 2;
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < time_clocks)
    {
      unsigned int end_clock = (unsigned int) clock();
      clock_offset = (clock_t)(end_clock - start_clock);
    }
  };

  auto clock_lambda_2 = [=] HOST_DEVICE (int idx) {
    m2[idx] = 1234;
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < time_clocks)
    {
      unsigned int end_clock = (unsigned int) clock();
      clock_offset = (clock_t)(end_clock - start_clock);
    }
  };

  auto clock_lambda_3 = [=] HOST_DEVICE (int idx) {
    float val = m1[idx];
    m1[idx] = val * val;
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < time_clocks)
    {
      unsigned int end_clock = (unsigned int) clock();
      clock_offset = (clock_t)(end_clock - start_clock);
    }
  };

#if defined(GPU)
  auto e1 = forall(dev1, 0, N, clock_lambda_1);
  dev2.wait_on(&e1);
  
  forall(dev1, 0, N, clock_lambda_3);

  forall(dev2, 0, N, clock_lambda_2);
  cudaDeviceSynchronize();
#else
  forall(dev1, 0, N, clock_lambda_1);
  forall(dev1, 0, N, clock_lambda_3);
  forall(dev2, 0, N, clock_lambda_2);
#endif

  // -----------------------------------------------------------------------
  

  std::cout << "------ M1 = (idx * 2) ^ 2 ------" << std::endl;
  for (int i = 0; i < 15; i++) {
    std::cout << m1[i] << std::endl;
  }

  std::cout << "------ M2 = 1234 ------" << std::endl;
  for (int i = 0; i < 15; i++) {
    std::cout << m2[i] << std::endl;
  }

  cudaProfilerStop();
  cudaDeviceReset();


  return 0;
}
