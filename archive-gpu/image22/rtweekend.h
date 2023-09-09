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

__device__ inline float random_float(curandState &rand_state) {
  // Returns a random real in [0,1).
  return 1.0f - curand_uniform(&rand_state);
}

__device__ inline float random_float(float min, float max,
                                     curandState &rand_state) {
  // Returns a random real in [min, max).
  return min + (max - min) * random_float(rand_state);
}

__device__ vec3 random_vec3(curandState &rand_state) {
  return vec3(random_float(rand_state), random_float(rand_state),
              random_float(rand_state));
}

__device__ vec3 random_vec3(float min, float max, curandState &rand_state) {
  return vec3(random_float(min, max, rand_state),
              random_float(min, max, rand_state),
              random_float(min, max, rand_state));
}

__device__ vec3 random_in_unit_sphere(curandState &rand_state) {
  while (true) {
    vec3 p = random_vec3(-1.0f, 1.0f, rand_state);
    if (p.length_squared() < 1.0f) {
      return p;
    }
  }
}

__device__ vec3 random_unit_vector(curandState &rand_state) {
  return unit_vector(random_in_unit_sphere(rand_state));
}

__device__ vec3 random_on_hemisphere(const vec3 &normal,
                                     curandState &rand_state) {
  vec3 on_unit_sphere = random_unit_vector(rand_state);
  return dot(on_unit_sphere, normal) > 0.0f ? on_unit_sphere : -on_unit_sphere;
}

__device__ inline vec3 random_in_unit_disk(curandState &rand_state) {
  while (true) {
    auto p = vec3(random_float(-1.0f, 1.0f, rand_state),
                  random_float(-1.0f, 1.0f, rand_state), 0.0f);
    if (p.length_squared() < 1.0f) {
      return p;
    }
  }
}

#endif