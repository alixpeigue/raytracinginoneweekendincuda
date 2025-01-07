#ifndef DATATYPEH
#define DATATYPEH

#include <cmath>

using DataType = float;

__host__ __device__ inline DataType float2datatype(float in) {
    return in;
}

__device__  inline DataType datatypesqrt(DataType in) {
    return sqrt(in);
}

__device__ inline DataType datatypetan(DataType in) {
    return sin(in)/cos(in);
}

#endif
