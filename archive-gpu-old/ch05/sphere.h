#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
  __device__ sphere() {}
  __device__ sphere(point3 cen, float r) : center(cen), radius(r){};
  __device__ virtual bool hit(const ray & r, float t_min, float t_max, hit_record & rec) const override;
  __device__ virtual ~sphere() {}

public:
  point3 center;
  float radius;
};

__device__ bool sphere::hit(const ray & r, float t_min, float t_max, hit_record & rec) const {
  vec3 oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius * radius;

  auto discriminant = half_b * half_b - a * c;
  if (discriminant < 0) {
    return false;
  }

  auto sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  auto root = (-half_b - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-half_b + sqrtd) / a;
    if (root < t_min || t_max < root)
      return false;
  }

  rec.t = root;
  rec.p = r.at(rec.t);
  rec.normal = (rec.p - center) / radius;

  return true;
}

#endif