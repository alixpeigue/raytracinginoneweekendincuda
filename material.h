#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"
#include "datatype.h"


__device__ DataType schlick(DataType cosine, DataType ref_idx) {
    DataType r0 = (float2datatype(1.0f)-ref_idx) / (float2datatype(1.0f)+ref_idx);
    r0 = r0*r0;
    return r0 + (float2datatype(1.0f)-r0)*float2datatype(pow((float2datatype(1.0f) - cosine),5.0f));
}

__device__ bool refract(const vec3& v, const vec3& n, DataType ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    DataType dt = dot(uv, n);
    DataType discriminant = float2datatype(1.0f) - ni_over_nt*ni_over_nt*(float2datatype(1.0f)-dt*dt);
    if (discriminant > float2datatype(0)) {
        refracted = ni_over_nt*(uv - n*dt) - n*datatypesqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= float2datatype(1.0f));
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - float2datatype(2.0f)*dot(v,n)*n;
}

class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);

             vec3 direction = target - rec.p;
             if (direction.length() < DataType(1e-8)) {
                 direction = rec.normal;
             }
             scattered = ray(rec.p, direction);

             attenuation = albedo;
             return true;
        }

        vec3 albedo;
};

class metal : public material {
    public:
        __device__ metal(const vec3& a, DataType f) : albedo(a) { if (f < float2datatype(1.0)) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > float2datatype(0.0f));
        }
        vec3 albedo;
        DataType fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(DataType ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered,
                         curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        DataType ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        DataType reflect_prob;
        DataType cosine;
        if (dot(r_in.direction(), rec.normal) > float2datatype(0.0f)) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = datatypesqrt(float2datatype(1.0f) - ref_idx*ref_idx*(float2datatype(1.0f)-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = float2datatype(1.0f) / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = float2datatype(1.0f);
        if (float2datatype(curand_uniform(local_rand_state)) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    DataType ref_idx;
};
#endif
