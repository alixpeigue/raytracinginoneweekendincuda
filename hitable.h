#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "datatype.h"

class material;

struct hit_record
{
    DataType t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, DataType t_min, DataType t_max, hit_record& rec) const = 0;
};

#endif
