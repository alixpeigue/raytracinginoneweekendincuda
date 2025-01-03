#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include <cuda_fp16.h>

class material;

struct hit_record {
  float t;
  vec3 p;
  vec3 normal;
  material *mat_ptr;
};

class hitable {
public:
  __device__ virtual bool hit(const ray &r, __half t_min, __half t_max,
                              hit_record &rec) const = 0;
};

#endif
