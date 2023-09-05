#ifndef INTERVAL_H
#define INTERVAL_H

#include "rtweekend.h"

class interval {
public:
  float min, max;

  __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

  __host__ __device__ interval(float _min, float _max) : min(_min), max(_max) {}

  __host__ __device__ bool contains(float x) const { return min <= x && x <= max; }

  __host__ __device__ bool surrounds(float x) const { return min < x && x < max; }

  static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif