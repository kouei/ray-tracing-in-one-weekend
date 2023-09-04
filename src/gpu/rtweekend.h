#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cuda_runtime.h>
#include <limits>

// Constants

constexpr const float infinity = std::numeric_limits<float>::infinity();
constexpr const float pi = 3.1415926535897932385f;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
  return degrees / 180.0f * pi;
}

#endif