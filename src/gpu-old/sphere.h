// Migration Completed

#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include <cmath>

class sphere : public hittable {
public:
  __device__ sphere() {}
  __device__ sphere(point3 cen, float r) : center(cen), radius(r){};
  __device__ sphere(point3 cen, float r, material * m) : center(cen), radius(r), mat_ptr(m) {};

  __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const override;
  __device__ virtual ~sphere() {}

public:
  point3 center;
  float radius;
  material * mat_ptr;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
  vec3 oc = r.origin() - this->center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - this->radius * this->radius;

  auto discriminant = half_b * half_b - a * c;
  if (discriminant < 0.0f) {
    return false;
  }

  auto sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  auto root = (-half_b - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-half_b + sqrtd) / a;
    if (root < t_min || t_max < root) {
      return false;
    }
  }

  rec.t = root;
  rec.p = r.at(rec.t);
  vec3 outward_normal = (rec.p - this->center) / radius;
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = this->mat_ptr;

  return true;
}

#endif