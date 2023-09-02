#ifndef CAMERAH
#define CAMERAH

#include <type_traits>
#include "ray.h"

class camera {
public:
  __device__ camera():
    lower_left_corner(-2.0f, -1.0f, -1.0f),
    horizontal(4.0f, 0.0f, 0.0f),
    vertical(0.0f, 2.0f, 0.0f),
    origin(0.0f, 0.0f, 0.0f)
  {}
  
  __device__ ray get_ray(float u, float v) {
    return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
  }

  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
};

using camera_ptr_t = std::add_pointer_t<camera>;

#endif