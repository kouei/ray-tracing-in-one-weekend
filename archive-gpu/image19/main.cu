#include "camera.h"
#include "color.h"
#include "cuda_utility.h"
#include "hittable.h"
#include "hittable_list.h"
#include "interval.h"
#include "material.h"
#include "ray.h"
#include "rtweekend.h"
#include "sphere.h"
#include "vec3.h"
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

__global__ void new_world(hittable_list *world) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  float R = cos(pi / 4.0f);

  material *material_left = new lambertian(color(0.0f, 0.0f, 1.0f));
  material *material_right = new lambertian(color(1.0f, 0.0f, 0.0f));

  new (world) hittable_list();
  world->objects = new hittable_ptr[2];
  world->add(new sphere(point3(-R, 0.0f, -1.0f), R, material_left));
  world->add(new sphere(point3(R, 0.0f, -1.0f), R, material_right));
}

__global__ void delete_world(hittable_list *world) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  for (size_t i = 0; i < world->objects_size; ++i) {
    delete world->objects[i];
  }
}

__global__ void new_rand_state(unsigned long long seed, camera *cam,
                               curandState *rand_states) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= cam->image_width) || (j >= cam->image_height)) {
    return;
  }

  int pixel_index = j * cam->image_width + i;
  // Each thread gets same seed, a different sequence number, no offset
  curand_init(seed, pixel_index, 0, &rand_states[pixel_index]);
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
  new_camera<<<1, 1>>>(cam);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Frame Buffer
  color *frame_buffer;
  int n_pixels = cam->image_width * cam->image_height;
  checkCudaErrors(
      cudaMallocManaged(&frame_buffer, n_pixels * sizeof(*frame_buffer)));

  // Choose Block Size and Thread Size
  const int n_thread_x = 16;
  const int n_thread_y = 16;
  int n_block_x = (cam->image_width + n_thread_x - 1) / n_thread_x;
  int n_block_y = (cam->image_height + n_thread_y - 1) / n_thread_y;
  dim3 blocks(n_block_x, n_block_y);
  dim3 threads(n_thread_x, n_thread_y);

  // Random State
  curandState *rand_states;
  checkCudaErrors(cudaMalloc(&rand_states, n_pixels * sizeof(*rand_states)));
  new_rand_state<<<blocks, threads>>>(time(nullptr), cam, rand_states);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Render
  std::clog << "Image Size = " << cam->image_width << " x " << cam->image_height
            << "\n";
  std::clog << "Samples Per Pixel = " << cam->samples_per_pixel << "\n";
  std::clog << "Block Dim (a x b threads) = " << n_thread_x << " x "
            << n_thread_y << "\n";

  auto start = std::chrono::high_resolution_clock::now();

  render<<<blocks, threads>>>(frame_buffer, cam, world, rand_states);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
  std::clog << "Time Cost = " << static_cast<int>(timer_in_ms.count() + 0.999f)
            << " ms\n";

  // Output Image
  output_image(cam, frame_buffer);

  // Cleanup Random State
  checkCudaErrors(cudaFree(rand_states));

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