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
    scattered =
        ray(rec.p, reflected + this->fuzz * random_unit_vector(rand_state));
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

class dielectric : public material {
public:
  __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

  __device__ bool scatter(const ray &r_in, const hit_record &rec,
                          color &attenuation, ray &scattered,
                          curandState &rand_state) const override {
    attenuation = color(1.0f, 1.0f, 1.0f);

    // We consider the "refractive index" of air as 1.0
    float refraction_ratio = rec.front_face ? (1.0f / this->ir) : this->ir;

    vec3 unit_direction = unit_vector(r_in.direction());
    float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    vec3 direction;

    if (cannot_refract) {
      direction = reflect(unit_direction, rec.normal);
    } else {
      direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    scattered = ray(rec.p, direction);
    return true;
  }

private:
  float ir; // Index of Refraction
};

#endif