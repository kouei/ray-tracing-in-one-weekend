#ifndef RAY_H
#define RAY_H

#include "vec3.h"
#include <cuda_runtime.h>

class ray {
  public:
    __host__ __device__ ray() {}

    __host__ __device__ ray(const point3 & origin, const vec3 & direction) : orig(origin), dir(direction) {}

    __host__ __device__ point3 origin() const  { return this->orig; }
    __host__ __device__ vec3 direction() const { return this->dir; }

    __host__ __device__  point3 at(float t) const {
        return this->orig + t * this->dir;
    }

  private:
    point3 orig;
    vec3 dir;
};

#endif