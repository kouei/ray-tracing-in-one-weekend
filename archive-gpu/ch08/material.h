#ifndef MATERIALH
#define MATERIALH

#include "hittable.h"
#include "ray.h"

__device__ vec3 random_in_unit_sphere(curandState * local_rand_state) {
  vec3 p;
  do {
    p = 2.0f * vec3::random(local_rand_state) - vec3(1.0f, 1.0f, 1.0f);
  } while (p.length_squared() >= 1.0f);
  return p;
}

class material {
public:
  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec,
                                  vec3 &attenuation, ray &scattered,
                                  curandState *local_rand_state) const = 0;
  __device__ ~material() {}
};

class lambertian : public material {
public:
  __device__ lambertian(const vec3 & a) : albedo(a) {}
  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec,
                                  vec3 &attenuation, ray &scattered,
                                  curandState *local_rand_state) const {
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target - rec.p);
    attenuation = albedo;
    return true;
  }

  __device__ virtual ~lambertian() {}

  vec3 albedo;
};

class metal : public material {
public:
  __device__ metal(const vec3 & a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec,
                                  vec3 &attenuation, ray &scattered,
                                  curandState *local_rand_state) const {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }

  __device__ virtual ~metal() {}

  vec3 albedo;
  float fuzz;
};

#endif