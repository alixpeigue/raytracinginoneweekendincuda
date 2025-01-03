#ifndef SPHEREH
#define SPHEREH

#include "cuda_fp16.h"
#include "hitable.h"

class sphere : public hitable {
public:
  __device__ sphere() {}
  __device__ sphere(vec3 cen, float r, material *m)
      : center(cen), radius(r), mat_ptr(m) {};
  __device__ virtual bool hit(const ray &r, __half tmin, __half tmax,
                              hit_record &rec) const;
  vec3 center;
  float radius;
  material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, __half t_min, __half t_max,
                            hit_record &rec) const {
  vec3 oc = r.origin() - center;
  __half a = dot(r.direction(), r.direction());
  __half b = dot(oc, r.direction());
  __half c = dot(oc, oc) - radius * radius;
  __half discriminant = b * b - a * c;
  if (discriminant > (__half)0.0f) {
    __half temp = (-b - hsqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
    temp = (-b + hsqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }
  return false;
}

#endif
