// Migration Completed

#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include <type_traits>

class material;

struct hit_record {
  point3 p;
  vec3 normal;
  material * mat_ptr;
  float t;
  bool front_face;

  __device__ inline void set_face_normal(const ray &r, const vec3 &outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0.0f;
    this->normal = front_face ? outward_normal : -outward_normal;
  }
};

class hittable {
public:
  __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;
  __device__ virtual ~hittable() {}
};

using hittable_ptr_t = std::add_pointer_t<hittable>;

#endif