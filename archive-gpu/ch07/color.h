#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

__host__ __device__ void normalize_by_sample(color & c, int sample_count) {
  float divisor = 1.0f / sample_count;
  c.e[0] *= divisor;
  c.e[1] *= divisor;
  c.e[2] *= divisor;
}

__host__ __device__ void gamma_correction(color & c) {
  c.e[0] = sqrt(c.e[0]);
  c.e[1] = sqrt(c.e[1]);
  c.e[2] = sqrt(c.e[2]);
}

#endif