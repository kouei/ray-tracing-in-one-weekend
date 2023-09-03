// Migration Completed

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable {
public:
  __device__ hittable_list() {}
  __device__ hittable_list(hittable_ptr_t * _objects, size_t _objects_size): objects(_objects), objects_size(_objects_size) {}

  __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const override;
  __device__ virtual ~hittable_list() {}

public:
  hittable_ptr_t * objects;
  size_t objects_size;
};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
  hit_record temp_rec;
  bool hit_anything = false;
  float closest_so_far = t_max;

  for(size_t i = 0; i < this->objects_size; ++i) {
    hittable_ptr_t object = this->objects[i];
    if (object->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}

#endif