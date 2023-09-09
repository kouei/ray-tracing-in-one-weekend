#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "interval.h"
#include "material.h"
#include "vec3.h"
#include <cuda_runtime.h>

class sphere : public hittable {
public:
  __device__ sphere(point3 _center, float _radius, material *_material)
      : center(_center), radius(_radius), mat(_material) {}

  __device__ bool hit(const ray &r, interval ray_t,
                      hit_record &rec) const override {
    vec3 oc = r.origin() - this->center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - this->radius * this->radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
      return false;
    }

    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
      root = (-half_b + sqrtd) / a;
      if (!ray_t.surrounds(root)) {
        return false;
      }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - this->center) / this->radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat = this->mat;
    return true;
  }

private:
  point3 center;
  float radius;
  material *mat;
};

#endif