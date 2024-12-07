#ifndef DATATYPEH
#define DATATYPEH

#include <cmath>
#include "cuda_fp16.h"

using DataType = __half;

__host__ __device__ DataType float2datatype(float in) {
    return in;
}

__device__ DataType datatypesqrt(DataType in) {
    return hsqrt(in);
}

#endif
