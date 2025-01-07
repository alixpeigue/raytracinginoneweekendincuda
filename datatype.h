#ifndef DATATYPEH
#define DATATYPEH

#include <cmath>
#include "cuda_bf16.h"

using DataType = __nv_bfloat16;

__host__ __device__ inline DataType float2datatype(float in) {
    return in;
}

__device__  inline DataType datatypesqrt(DataType in) {
    return hsqrt(in);
}

__device__ inline DataType datatypetan(DataType in) {
    return hsin(in)/hcos(in);
}

#endif
