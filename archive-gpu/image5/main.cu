#include "color.h"
#include "cuda_utility.h"
#include "hittable.h"
#include "hittable_list.h"
#include "ray.h"
#include "rtweekend.h"
#include "sphere.h"
#include "vec3.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

__device__ color ray_color(const ray &r, const hittable &world) {
  hit_record rec;
  if (world.hit(r, 0.0f, infinity, rec)) {
    return 0.5f * (rec.normal + color(1.0f, 1.0f, 1.0f));
  }

  vec3 unit_direction = unit_vector(r.direction());
  float a = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3 *frame_buffer, int image_width, int image_height,
                       vec3 pixel_delta_u, vec3 pixel_delta_v,
                       point3 pixel00_loc, point3 camera_center,
                       hittable_list *world) {

  int image_x = threadIdx.x + blockIdx.x * blockDim.x;
  int image_y = threadIdx.y + blockIdx.y * blockDim.y;
  if (image_x >= image_width || image_y >= image_height) {
    return;
  }

  int pixel_index = image_y * image_width + image_x;

  auto pixel_center =
      pixel00_loc + (image_x * pixel_delta_u) + (image_y * pixel_delta_v);
  auto ray_direction = pixel_center - camera_center;
  ray r(camera_center, ray_direction);

  color pixel_color = ray_color(r, world[0]);
  frame_buffer[pixel_index] = pixel_color;
}

__global__ void create_world(hittable_list *world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (world) hittable_list();
    world->objects = new hittable_ptr[2];
    world->add(new sphere(point3(0.0f, 0.0f, -1.0f), 0.5f));
    world->add(new sphere(point3(0.0f, -100.5f, -1.0f), 100.0f));
  }
}

hittable_list *create_world_cpu() {
  hittable_list *world;
  checkCudaErrors(cudaMalloc(&world, sizeof(*world)));
  create_world<<<1, 1>>>(world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  return world;
}

int main() {

  // Image
  float aspect_ratio = 16.0f / 9.0f;
  int image_width = 1920;

  // Calculate the image height, and ensure that it's at least 1.
  int image_height = static_cast<int>(image_width / aspect_ratio);
  image_height = (image_height < 1) ? 1 : image_height;

  const int samples_per_pixel = 500;
  const int n_thread_x = 16;
  const int n_thread_y = 16;

  std::clog << "Image Size = " << image_width << "x" << image_height << "\n";
  std::clog << "Samples Per Pixel = " << samples_per_pixel << "\n";
  std::clog << "Block Dim (a x b threads) = " << n_thread_x << "x" << n_thread_y
            << "\n";

  // World

  hittable_list *world = create_world_cpu();

  // Camera

  float focal_length = 1.0f;
  float viewport_height = 2.0f;
  float viewport_width =
      viewport_height * ((float)(image_width) / image_height);
  point3 camera_center = point3(0.0f, 0.0f, 0.0f);

  // Calculate the vectors across the horizontal and down the vertical viewport
  // edges.
  vec3 viewport_u = vec3(viewport_width, 0.0f, 0.0f);
  vec3 viewport_v = vec3(0.0f, -viewport_height, 0.0f);

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  vec3 pixel_delta_u = viewport_u / (float)(image_width);
  vec3 pixel_delta_v = viewport_v / (float)(image_height);

  // Calculate the location of the upper left pixel.
  point3 viewport_upper_left = camera_center - vec3(0.0f, 0.0f, focal_length) -
                               viewport_u / 2.0f - viewport_v / 2.0f;

  point3 pixel00_loc =
      viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

  // Allocate Frame Buffer
  int n_pixels = image_width * image_height;
  vec3 *frame_buffer;
  size_t frame_buffer_size = n_pixels * sizeof(*frame_buffer);

  checkCudaErrors(cudaMallocManaged(&frame_buffer, frame_buffer_size));

  // Render

  int n_block_x = (image_width + n_thread_x - 1) / n_thread_x;
  int n_block_y = (image_height + n_thread_y - 1) / n_thread_y;
  dim3 blocks(n_block_x, n_block_y);
  dim3 threads(n_thread_x, n_thread_y);

  auto start = std::chrono::high_resolution_clock::now();

  render<<<blocks, threads>>>(frame_buffer, image_width, image_height,
                              pixel_delta_u, pixel_delta_v, pixel00_loc,
                              camera_center, world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
  std::clog << "Time Cost = " << static_cast<int>(timer_in_ms.count() + 0.999f)
            << " ms\n";

  // Output Image

  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int image_y = 0; image_y < image_height; ++image_y) {
    for (int image_x = 0; image_x < image_width; ++image_x) {
      int pixel_index = image_y * image_width + image_x;
      vec3 pixel = frame_buffer[pixel_index];

      int ir = static_cast<int>(255.999 * pixel.x());
      int ig = static_cast<int>(255.999 * pixel.y());
      int ib = static_cast<int>(255.999 * pixel.z());

      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
}