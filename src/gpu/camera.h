// Migration Completed

#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"
#include <cmath>
#include <curand_kernel.h>

class camera {
public:
  __device__ camera(point3 lookfrom, point3 lookat, vec3 vup,
                    float vfov, // vertical field-of-view in degrees
                    float aspect_ratio, float aperture, float focus_dist) {
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta / 2.0f);
    auto viewport_height = 2.0f * h;
    auto viewport_width = aspect_ratio * viewport_height;

    this->w = unit_vector(lookfrom - lookat);
    this->u = unit_vector(cross(vup, this->w));
    this->v = cross(this->w, this->u);

    this->origin = lookfrom;
    this->horizontal = focus_dist * viewport_width * this->u;
    this->vertical = focus_dist * viewport_height * this->v;
    this->lower_left_corner = this->origin - this->horizontal / 2.0f - this->vertical / 2.0f - focus_dist * this->w;

    this->lens_radius = aperture / 2.0f;
  }

  __device__ ray get_ray(float s, float t, curandState * local_rand_state) const {
    vec3 rd = this->lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = this->u * rd.x() + this->v * rd.y();

    return ray(this->origin + offset, this->lower_left_corner + s * this->horizontal + t * this->vertical - this->origin - offset);
  }

private:
  point3 origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  float lens_radius;
};

#endif