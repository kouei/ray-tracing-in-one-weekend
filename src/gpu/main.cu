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

__global__ void new_world(hittable_list *world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (world) hittable_list();
    world->objects = new hittable_ptr[2];
    world->add(new sphere(point3(0.0f, 0.0f, -1.0f), 0.5f));
    world->add(new sphere(point3(0.0f, -100.5f, -1.0f), 100.0f));
  }
}

__global__ void delete_world(hittable_list *world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (size_t i = 0; i < world->objects_size; ++i) {
      delete world->objects[i];
    }
  }
}

int main() {

  // World
  hittable_list *world;
  checkCudaErrors(cudaMalloc(&world, sizeof(*world)));
  new_world<<<1, 1>>>(world);
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

  // Frame Buffer
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
  output_image(cam, frame_buffer);

  // Cleanup Frame Buffer
  checkCudaErrors(cudaFree(frame_buffer));

  // Cleanup Camera
  checkCudaErrors(cudaFree(cam));

  // Cleanup World
  delete_world<<<1, 1>>>(world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(world));
}