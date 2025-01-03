#ifndef VEC3HALFH
#define VEC3HALFH

#include <cuda_fp16.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

class vec3_half {

public:
  __host__ __device__ vec3_half() {}
  __host__ __device__ vec3_half(float e0, float e1, float e2) {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
  }
  __host__ __device__ inline float x() const { return e[0]; }
  __host__ __device__ inline float y() const { return e[1]; }
  __host__ __device__ inline float z() const { return e[2]; }
  __host__ __device__ inline float r() const { return e[0]; }
  __host__ __device__ inline float g() const { return e[1]; }
  __host__ __device__ inline float b() const { return e[2]; }

  __host__ __device__ inline const vec3_half &operator+() const {
    return *this;
  }
  __host__ __device__ inline vec3_half operator-() const {
    return vec3_half(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ inline __half operator[](int i) const { return e[i]; }
  __host__ __device__ inline __half &operator[](int i) { return e[i]; };

  __host__ __device__ inline vec3_half &operator+=(const vec3_half &v2);
  __host__ __device__ inline vec3_half &operator-=(const vec3_half &v2);
  __host__ __device__ inline vec3_half &operator*=(const vec3_half &v2);
  __host__ __device__ inline vec3_half &operator/=(const vec3_half &v2);
  __host__ __device__ inline vec3_half &operator*=(const float t);
  __host__ __device__ inline vec3_half &operator/=(const float t);

  __device__ inline float length() const {
    return hsqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  }
  __device__ inline float squared_length() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  __device__ inline void make_unit_vector();

  __half e[3];
};

__device__ inline void vec3_half::make_unit_vector() {
  float k = (__half)1.0 / hsqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
}

__host__ __device__ inline vec3_half operator+(const vec3_half &v1,
                                               const vec3_half &v2) {
  return vec3_half(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3_half operator-(const vec3_half &v1,
                                               const vec3_half &v2) {
  return vec3_half(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3_half operator*(const vec3_half &v1,
                                               const vec3_half &v2) {
  return vec3_half(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3_half operator/(const vec3_half &v1,
                                               const vec3_half &v2) {
  return vec3_half(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3_half operator*(__half t, const vec3_half &v) {
  return vec3_half(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3_half operator/(vec3_half v, __half t) {
  return vec3_half(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3_half operator*(const vec3_half &v, __half t) {
  return vec3_half(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const vec3_half &v1, const vec3_half &v2) {
  return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3_half cross(const vec3_half &v1,
                                           const vec3_half &v2) {
  return vec3_half((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                   (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                   (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline vec3_half &
vec3_half::operator+=(const vec3_half &v) {
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

__host__ __device__ inline vec3_half &
vec3_half::operator*=(const vec3_half &v) {
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

__host__ __device__ inline vec3_half &
vec3_half::operator/=(const vec3_half &v) {
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

__host__ __device__ inline vec3_half &
vec3_half::operator-=(const vec3_half &v) {
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

__host__ __device__ inline vec3_half &vec3_half::operator*=(const float t) {
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

__host__ __device__ inline vec3_half &vec3_half::operator/=(const float t) {
  float k = 1.0 / t;

  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

__device__ inline vec3_half unit_vector(vec3_half v) { return v / v.length(); }

#endif
