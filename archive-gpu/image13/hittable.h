#ifndef HITTABLE_H
#define HITTABLE_H

#include "interval.h"
#include "ray.h"
#include <cuda_runtime.h>

class material;

class hit_record {
public:
  point3 p;
  vec3 normal;
  material * mat;
  float t;
  bool front_face;

  __host__ __device__ void set_face_normal(const ray &r,
                                           const vec3 &outward_normal) {
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.
    // this->normal will always point against the ray.

    this->front_face = dot(r.direction(), outward_normal) < 0.0f;
    this->normal = this->front_face ? outward_normal : -outward_normal;
  }
};

class hittable {
public:
  __host__ __device__ virtual ~hittable() {}
  __host__ __device__ virtual bool
  hit(const ray &r, interval ray_t, hit_record &rec) const = 0;
};

typedef hittable *hittable_ptr;

#endif