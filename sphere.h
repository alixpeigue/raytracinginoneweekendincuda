#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "datatype.h"

class sphere: public hitable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, DataType r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, DataType tmin, DataType tmax, hit_record& rec) const;
        vec3 center;
        DataType radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, DataType t_min, DataType t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    DataType a = dot(r.direction(), r.direction());
    DataType b = dot(oc, r.direction());
    DataType c = dot(oc, oc) - radius*radius;
    DataType discriminant = b*b - a*c;
    if (discriminant > float2datatype(0)) {
        DataType temp = (-b - datatypesqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + datatypesqrt(discriminant)) / a;
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
