#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include "vec3.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits>

// Constants

constexpr const float infinity = std::numeric_limits<float>::infinity();
constexpr const float pi = 3.1415926535897932385f;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
  return degrees / 180.0f * pi;
}

__device__ inline float random_float(curandState *local_rand_state) {
  // Returns a random real in [0,1).
  return 1.0f - curand_uniform(local_rand_state);
}

__device__ inline float random_float(float min, float max,
                                     curandState *local_rand_state) {
  // Returns a random real in [min, max).
  return min + (max - min) * random_float(local_rand_state);
}

__device__ vec3 random_vec3(curandState *local_rand_state) {
  return vec3(random_float(local_rand_state), random_float(local_rand_state),
              random_float(local_rand_state));
}

__device__ vec3 random_vec3(float min, float max,
                            curandState *local_rand_state) {
  return vec3(random_float(min, max, local_rand_state),
              random_float(min, max, local_rand_state),
              random_float(min, max, local_rand_state));
}

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
  while (true) {
    vec3 p = random_vec3(-1.0f, 1.0f, local_rand_state);
    if (p.length_squared() < 1.0f) {
      return p;
    }
  }
}

__device__ vec3 random_unit_vector(curandState *local_rand_state) {
  return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ vec3 random_on_hemisphere(const vec3 &normal,
                                     curandState *local_rand_state) {
  vec3 on_unit_sphere = random_unit_vector(local_rand_state);
  return dot(on_unit_sphere, normal) > 0.0f ? on_unit_sphere : -on_unit_sphere;
}

#endif