#include "camera.h"
#include "color.h"
#include "cuda_utility.h"
#include "hittable.h"
#include "hittable_list.h"
#include "interval.h"
#include "ray.h"
#include "rtweekend.h"
#include "sphere.h"
#include "vec3.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

__global__ void create_world(hittable_list *world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (world) hittable_list();
    world->objects = new hittable_ptr[2];
    world->add(new sphere(point3(0.0f, 0.0f, -1.0f), 0.5f));
    world->add(new sphere(point3(0.0f, -100.5f, -1.0f), 100.0f));
  }
}

int main() {

  // World

  hittable_list *world;
  checkCudaErrors(cudaMalloc(&world, sizeof(*world)));
  create_world<<<1, 1>>>(world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Camera

  camera *cam;
  checkCudaErrors(cudaMallocManaged(&cam, sizeof(*cam)));
  initialize<<<1, 1>>>(cam);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  const int samples_per_pixel = 500;
  const int n_thread_x = 16;
  const int n_thread_y = 16;

  std::clog << "Image Size = " << cam->image_width << "x" << cam->image_height
            << "\n";
  std::clog << "Samples Per Pixel = " << samples_per_pixel << "\n";
  std::clog << "Block Dim (a x b threads) = " << n_thread_x << "x" << n_thread_y
            << "\n";

  // Allocate Frame Buffer

  color *frame_buffer;
  int n_pixels = cam->image_width * cam->image_height;
  checkCudaErrors(
      cudaMallocManaged(&frame_buffer, n_pixels * sizeof(*frame_buffer)));

  // Render

  int n_block_x = (cam->image_width + n_thread_x - 1) / n_thread_x;
  int n_block_y = (cam->image_height + n_thread_y - 1) / n_thread_y;
  dim3 blocks(n_block_x, n_block_y);
  dim3 threads(n_thread_x, n_thread_y);

  auto start = std::chrono::high_resolution_clock::now();

  render<<<blocks, threads>>>(frame_buffer, cam, world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
  std::clog << "Time Cost = " << static_cast<int>(timer_in_ms.count() + 0.999f)
            << " ms\n";

  // Output Image

  std::cout << "P3\n"
            << cam->image_width << ' ' << cam->image_height << "\n255\n";

  for (int image_y = 0; image_y < cam->image_height; ++image_y) {
    for (int image_x = 0; image_x < cam->image_width; ++image_x) {
      int pixel_index = image_y * cam->image_width + image_x;
      color pixel = frame_buffer[pixel_index];

      int ir = static_cast<int>(255.999 * pixel.x());
      int ig = static_cast<int>(255.999 * pixel.y());
      int ib = static_cast<int>(255.999 * pixel.z());

      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
}