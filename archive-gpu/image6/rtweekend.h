#ifndef RTWEEKEND_H
#define RTWEEKEND_H

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

#endif