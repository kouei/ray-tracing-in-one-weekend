#ifndef CAMERA_H
#define CAMERA_H

#include "color.h"
#include "cuda_utility.h"
#include "hittable.h"
#include "material.h"
#include "rtweekend.h"
#include <cuda_runtime.h>

class camera {
public:
  float aspect_ratio;
  int image_width;
  int image_height;
  int samples_per_pixel;
  int max_depth; // Maximum number of ray bounces into scene

  float vfov; // Vertical view angle (field of view)

private:
  point3 center;      // Camera center
  point3 pixel00_loc; // Location of pixel 0, 0
  vec3 pixel_delta_u; // Offset to pixel to the right
  vec3 pixel_delta_v; // Offset to pixel below

public:
  __global__ friend void new_camera(camera *cam);

  __global__ friend void render(color *frame_buffer, camera *cam,
                                hittable *world, curandState *rand_state);

  __device__ friend vec3 pixel_sample_square(camera &cam,
                                             curandState &rand_state);

  __device__ friend ray get_ray(camera &cam, int i, int j,
                                curandState &rand_state);
};

__global__ void new_camera(camera *cam) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  cam->max_depth = 50;
  cam->samples_per_pixel = 10;
  cam->aspect_ratio = 16.0f / 9.0f;
  cam->image_width = 1920;
  cam->image_height = static_cast<int>(cam->image_width / cam->aspect_ratio);
  cam->image_height = (cam->image_height < 1) ? 1 : cam->image_height;

  cam->vfov = 90.0f;

  cam->center = point3(0.0f, 0.0f, 0.0f);

  // Determine viewport dimensions.
  float focal_length = 1.0f;
  float theta = degrees_to_radians(cam->vfov);
  float h = tan(theta / 2.0f);
  float viewport_height = 2.0f * h * focal_length;
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

__device__ color ray_color(const ray &r, const hittable &world,
                           curandState &rand_state, int max_depth) {
  hit_record rec;
  ray cur_ray = r;
  color cur_attenuation = color(1.0f, 1.0f, 1.0f);
  for (int i = 0; i < max_depth; ++i) {
    if (!world.hit(cur_ray, interval(0.001f, infinity), rec)) {
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float a = 0.5f * (unit_direction.y() + 1.0f);
      color output_color =
          (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
      return cur_attenuation * output_color;
    }

    ray scattered;
    color attenuation;
    if (!rec.mat->scatter(cur_ray, rec, attenuation, scattered, rand_state)) {
      return color(0.0f, 0.0f, 0.0f);
    }

    cur_attenuation *= attenuation;
    cur_ray = scattered;
  }

  // Reach max depth
  return color(0.0f, 0.0f, 0.0f);
}

__device__ vec3 pixel_sample_square(camera &cam, curandState &rand_state) {
  // Returns a random point in the square surrounding a pixel at the origin.
  auto px = -0.5f + random_float(rand_state);
  auto py = -0.5f + random_float(rand_state);
  return (px * cam.pixel_delta_u) + (py * cam.pixel_delta_v);
}

__device__ ray get_ray(camera &cam, int i, int j, curandState &rand_state) {
  // Get a randomly sampled camera ray for the pixel at location i,j.

  auto pixel_center =
      cam.pixel00_loc + (i * cam.pixel_delta_u) + (j * cam.pixel_delta_v);
  auto pixel_sample = pixel_center + pixel_sample_square(cam, rand_state);

  auto ray_origin = cam.center;
  auto ray_direction = pixel_sample - ray_origin;

  return ray(ray_origin, ray_direction);
}

__global__ void render(color *frame_buffer, camera *cam, hittable *world,
                       curandState *rand_states) {

  int image_x = threadIdx.x + blockIdx.x * blockDim.x;
  int image_y = threadIdx.y + blockIdx.y * blockDim.y;
  if (image_x >= cam->image_width || image_y >= cam->image_height) {
    return;
  }

  int pixel_index = image_y * cam->image_width + image_x;

  auto pixel_center = cam->pixel00_loc + (image_x * cam->pixel_delta_u) +
                      (image_y * cam->pixel_delta_v);
  auto ray_direction = pixel_center - cam->center;

  color pixel_color = color(0.0f, 0.0f, 0.0f);
  curandState rand_state = rand_states[pixel_index];
  for (int sample = 0; sample < cam->samples_per_pixel; ++sample) {
    ray r = get_ray(cam[0], image_x, image_y, rand_state);
    pixel_color += ray_color(r, world[0], rand_state, cam->max_depth);
  }

  rand_states[pixel_index] = rand_state;
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
      write_color(std::cout, pixel, cam->samples_per_pixel);
    }
  }
}

#endif