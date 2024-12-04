#ifndef DATATYPEH
#define DATATYPEH

#include <cmath>

using DataType = float;

__host__ __device__ DataType float2datatype(float in) {
    return in;
}

__host__ __device__ DataType datatypesqrt(DataType in) {
    return std::sqrt(in);
}

#endif
