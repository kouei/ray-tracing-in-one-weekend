#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "hittable.h"
#include "ray.h"

class hit_record;

class material {
public:
  __device__ virtual ~material() {}
  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec,
                                  color &attenuation, ray &scattered,
                                  curandState &rand_state) const = 0;
};

class lambertian : public material {
public:
  __device__ lambertian(const color &a) : albedo(a) {}

  __device__ bool scatter(const ray &r_in, const hit_record &rec,
                          color &attenuation, ray &scattered,
                          curandState &rand_state) const override {
    auto scatter_direction = rec.normal + random_unit_vector(rand_state);
    // Catch degenerate scatter direction
    if (scatter_direction.near_zero()) {
      scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction);
    attenuation = this->albedo;
    return true;
  }

private:
  color albedo;
};

class metal : public material {
public:
  __device__ metal(const color &a, float f)
      : albedo(a), fuzz(f < 1.0f ? f : 1) {}

  __device__ bool scatter(const ray &r_in, const hit_record &rec,
                          color &attenuation, ray &scattered,
                          curandState &rand_state) const override {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + this->fuzz * random_unit_vector(rand_state));
    attenuation = this->albedo;
    
    // If the dot is < 0, then the scattered ray is inside the surface.
    // In this case, we consider the scattered ray as absorbed by the surface.
    // Hence, no scattered ray will be generated in this situation.
    return dot(scattered.direction(), rec.normal) > 0;
  }

private:
  color albedo;
  float fuzz;
};

#endif