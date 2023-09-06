#ifndef CAMERA_H
#define CAMERA_H

#include "color.h"
#include "cuda_utility.h"
#include "hittable.h"
#include "rtweekend.h"
#include <cuda_runtime.h>

class camera {
public:
  float aspect_ratio;
  int image_width;
  int image_height;

private:
  point3 center;      // Camera center
  point3 pixel00_loc; // Location of pixel 0, 0
  vec3 pixel_delta_u; // Offset to pixel to the right
  vec3 pixel_delta_v; // Offset to pixel below

public:
  __global__ friend void initialize(camera *cam);
  __global__ friend void render(color *frame_buffer, camera *cam,
                                hittable *world);
};

__global__ void initialize(camera *cam) {
  if (!(threadIdx.x == 0 && blockIdx.x == 0)) {
    return;
  }

  cam->aspect_ratio = 16.0f / 9.0f;
  cam->image_width = 1920;
  cam->image_height = static_cast<int>(cam->image_width / cam->aspect_ratio);
  cam->image_height = (cam->image_height < 1) ? 1 : cam->image_height;

  cam->center = point3(0.0f, 0.0f, 0.0f);

  // Determine viewport dimensions.
  float focal_length = 1.0f;
  float viewport_height = 2.0f;
  float viewport_width =
      viewport_height * ((float)(cam->image_width) / cam->image_height);

  // Calculate the vectors across the horizontal and down the vertical
  // viewport edges.
  vec3 viewport_u = vec3(viewport_width, 0.0f, 0.0f);
  vec3 viewport_v = vec3(0.0f, -viewport_height, 0.0f);

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  cam->pixel_delta_u = viewport_u / (float)(cam->image_width);
  cam->pixel_delta_v = viewport_v / (float)(cam->image_height);

  // Calculate the location of the upper left pixel.
  point3 viewport_upper_left = cam->center - vec3(0.0f, 0.0f, focal_length) -
                               viewport_u / 2.0f - viewport_v / 2.0f;
  cam->pixel00_loc =
      viewport_upper_left + 0.5f * (cam->pixel_delta_u + cam->pixel_delta_v);
}

__device__ color ray_color(const ray &r, const hittable &world) {
  hit_record rec;
  if (world.hit(r, interval(0.0f, infinity), rec)) {
    return 0.5f * (rec.normal + color(1.0f, 1.0f, 1.0f));
  }

  vec3 unit_direction = unit_vector(r.direction());
  float a = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
}

__global__ void render(color *frame_buffer, camera *cam, hittable *world) {

  int image_x = threadIdx.x + blockIdx.x * blockDim.x;
  int image_y = threadIdx.y + blockIdx.y * blockDim.y;
  if (image_x >= cam->image_width || image_y >= cam->image_height) {
    return;
  }

  int pixel_index = image_y * cam->image_width + image_x;

  auto pixel_center = cam->pixel00_loc + (image_x * cam->pixel_delta_u) +
                      (image_y * cam->pixel_delta_v);
  auto ray_direction = pixel_center - cam->center;
  ray r(cam->center, ray_direction);

  color pixel_color = ray_color(r, world[0]);
  frame_buffer[pixel_index] = pixel_color;
}

__host__ void output_image(camera *cam, color *frame_buffer) {
  int image_width = cam->image_width;
  int image_height = cam->image_height;

  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      int pixel_index = y * image_width + x;
      color pixel = frame_buffer[pixel_index];

      int ir = static_cast<int>(255.999f * pixel.x());
      int ig = static_cast<int>(255.999f * pixel.y());
      int ib = static_cast<int>(255.999f * pixel.z());

      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
}

#endif