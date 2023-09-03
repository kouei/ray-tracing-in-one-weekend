// Migration Completed

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <curand_kernel.h>
#include <limits>

constexpr const float infinity = std::numeric_limits<float>::infinity();
constexpr const float pi = 3.1415926535897932385f;

__host__ __device__ inline float degrees_to_radians(float degrees) {
  return degrees * pi / 180.0f;
}

// curand_uniform() returns value in (0.0f, 1.0f],
// Hence write "1.0f - curand_uniform()" instead,
// to convert it to [0.0f, 1.0f)
__device__ inline float random_double(curandState * local_rand_state) {
  return 1.0f - curand_uniform(local_rand_state);
}

// Returns a random real in [min,max).
__device__ inline float random_double(float min, float max, curandState * local_rand_state) {
  return min + (max - min) * random_double(local_rand_state);
}

__host__ __device__ inline float clamp(float x, float min, float max) {
  if (x < min) {
    return min;
  } else if (x > max) {
    return max;
  } else {
    return x;
  }
}

#endif