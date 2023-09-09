#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "interval.h"
#include <cuda_runtime.h>

class hittable_list : public hittable {
public:
  hittable_ptr *objects;
  size_t objects_size;
  size_t objects_capacity;

  __device__ hittable_list() : objects(nullptr), objects_size(0), objects_capacity(0) {}

  __device__ void add(hittable_ptr object) {
    if (!this->objects) {
      this->objects = new hittable_ptr[1];
      this->objects_capacity = 1;
    }

    if (this->objects_size >= this->objects_capacity) {
      size_t new_objects_capacity = this->objects_capacity * 2;
      hittable_ptr *new_objects = new hittable_ptr[new_objects_capacity];
      for (size_t i = 0; i < this->objects_size; ++i) {
        new_objects[i] = this->objects[i];
      }

      delete [] this->objects;
      this->objects = new_objects;
      this->objects_capacity = new_objects_capacity;
    }
    
    this->objects[this->objects_size++] = object;
  }

  __device__ bool hit(const ray &r, interval ray_t,
                               hit_record &rec) const override {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (size_t i = 0; i < this->objects_size; ++i) {
      hittable_ptr object = this->objects[i];
      if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
};

#endif