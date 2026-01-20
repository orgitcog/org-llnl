#pragma once

enum class MemorySpace { CPU, GPU, UNIFIED };

#ifdef __CUDACC__

#include <cuda.h>

#else

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#define __forceinline__
#endif

#endif