#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include <type_traits>

class material;

struct hit_record {
  float t;
  point3 p;
  vec3 normal;
  material * mat_ptr;
};

class hittable {
public:
  __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
  __device__ virtual ~hittable() {}
};

using hittable_ptr_t = std::add_pointer_t<hittable>;

#endif