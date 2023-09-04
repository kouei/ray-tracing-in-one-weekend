#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include <cuda_runtime.h>

class hittable_list : public hittable {
public:
  hittable_ptr *objects;
  size_t objects_size;

  __host__ __device__ hittable_list() : objects(nullptr), objects_size(0) {}

  __host__ __device__ void add(hittable_ptr object) {
    this->objects[this->objects_size++] = object;
  }

  __host__ __device__ bool hit(const ray &r, float ray_tmin, float ray_tmax,
                               hit_record &rec) const override {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_tmax;

    for (size_t i = 0; i < this->objects_size; ++i) {
      hittable_ptr object = this->objects[i];
      if (object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
};

#endif