#ifndef DATATYPEH
#define DATATYPEH

#include <cmath>
#include "cuda_fp16.h"

using DataType = __half;

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
