#include "camera.h"
#include "color.h"
#include "cuda_utility.h"
#include "hittable.h"
#include "hittable_list.h"
#include "interval.h"
#include "material.h"
#include "material_list.h"
#include "ray.h"
#include "rtweekend.h"
#include "sphere.h"
#include "vec3.h"
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

__global__ void new_world(hittable_list *world, material_list *materials,
                          unsigned long long seed) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  curandState rand_state;
  curand_init(seed, 0, 0, &rand_state);

  new (materials) material_list();

  material *ground_material = new lambertian(color(0.5f, 0.5f, 0.5f));
  materials->add(ground_material);

  new (world) hittable_list();
  world->add(
      new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = random_float(rand_state);
      point3 center(a + 0.9f * random_float(rand_state), 0.2f,
                    b + 0.9f * random_float(rand_state));

      if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
        material *sphere_material;

        if (choose_mat < 0.8f) {
          // diffuse
          color albedo = random_vec3(rand_state) * random_vec3(rand_state);
          sphere_material = new lambertian(albedo);
        } else if (choose_mat < 0.95f) {
          // metal
          color albedo = random_vec3(0.5f, 1.0f, rand_state);
          float fuzz = random_float(0.0f, 0.5f, rand_state);
          sphere_material = new metal(albedo, fuzz);
        } else {
          // glass
          sphere_material = new dielectric(1.5f);
        }

        world->add(new sphere(center, 0.2f, sphere_material));
      }
    }
  }

  material *material1 = new dielectric(1.5f);
  materials->add(material1);
  world->add(new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, material1));

  material *material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
  materials->add(material2);
  world->add(new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2));

  material *material3 = new metal(color(0.7f, 0.6f, 0.5f), 0.0f);
  materials->add(material3);
  world->add(new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, material3));
}

__global__ void delete_world(hittable_list *world, material_list *materials) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  world->~hittable_list();
  materials->~material_list();
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

  const unsigned long long seed = time(nullptr);

  // World
  hittable_list *world;
  checkCudaErrors(cudaMalloc(&world, sizeof(*world)));
  material_list *materials;
  checkCudaErrors(cudaMalloc(&materials, sizeof(*materials)));
  new_world<<<1, 1>>>(world, materials, seed);
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
  new_rand_state<<<blocks, threads>>>(seed, cam, rand_states);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Render
  std::clog << "Image Size = " << cam->image_width << " x " << cam->image_height
            << "\n";
  std::clog << "Samples Per Pixel = " << cam->samples_per_pixel << "\n";
  std::clog << "Block Dim (a x b threads) = " << n_thread_x << " x "
            << n_thread_y << "\n";
  std::clog << "Random Seed = " << seed << "\n";

  auto start = std::chrono::high_resolution_clock::now();

  render<<<blocks, threads>>>(frame_buffer, cam, world, rand_states);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
  auto time_cost_in_ms = static_cast<int>(timer_in_ms.count() + 0.999f);
  auto time_cost_in_sec = (time_cost_in_ms + 999) / 1000;
  std::clog << "Time Cost (ms) = " << time_cost_in_ms << " ms\n";
  std::clog << "Time Cost (sec) = " << time_cost_in_sec << " sec\n";

  // Output Image
  output_image(cam, frame_buffer);

  // Cleanup Random State
  checkCudaErrors(cudaFree(rand_states));

  // Cleanup Frame Buffer
  checkCudaErrors(cudaFree(frame_buffer));

  // Cleanup Camera
  checkCudaErrors(cudaFree(cam));

  // Cleanup World
  delete_world<<<1, 1>>>(world, materials);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(world));
  checkCudaErrors(cudaFree(materials));
}