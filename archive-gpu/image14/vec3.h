#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

using std::sqrt;

class vec3 {
public:
  float e[3];

  __host__ __device__ vec3() : e{0.0f, 0.0f, 0.0f} {}
  __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

  __host__ __device__ float x() const { return e[0]; }
  __host__ __device__ float y() const { return e[1]; }
  __host__ __device__ float z() const { return e[2]; }

  __host__ __device__ vec3 operator-() const {
    return vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ float operator[](int i) const { return e[i]; }
  __host__ __device__ float &operator[](int i) { return e[i]; }

  __host__ __device__ vec3 &operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  __host__ __device__ vec3 &operator*=(float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }

  __host__ __device__ vec3 &operator*=(const vec3 &v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
  }

  __host__ __device__ vec3 &operator/=(float t) { return *this *= 1.0f / t; }

  __host__ __device__ float length() const { return sqrt(length_squared()); }

  __host__ __device__ float length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  __host__ __device__ bool near_zero() const {
    // Return true if the vector is close to zero in all dimensions.
    float s = 1e-8f;
    return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
  }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the
// code.
using point3 = vec3;

// Vector Utility Functions

__host__ inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
  return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
  return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
  return (1.0f / t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3 &v) {
  return v / v.length();
}

__host__ __device__ vec3 reflect(const vec3 &v, const vec3 &n) {
  return v - 2.0f * dot(v, n) * n;
}

#endif