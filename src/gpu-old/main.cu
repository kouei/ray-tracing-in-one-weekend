// Migration Completed

#include "camera.h"
#include "color.h"
#include "cuda_utility.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

#include <chrono>
#include <ctime>
#include <curand_kernel.h>
#include <iostream>

__device__ color ray_color(const ray &r, hittable_ptr_t world, int depth, curandState * local_rand_state) {
  hit_record rec;
  ray current_ray = r;
  color current_attenuation = color(1.0f, 1.0f, 1.0f);

  // If we've exceeded the ray bounce limit, no more light is gathered.
  for (int i = 0; i < depth; ++i) {
    if (world->hit(current_ray, 0.001f, infinity, rec)) {
      ray scattered;
      color attenuation;
      if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state)) {
        current_attenuation *= attenuation;
        current_ray = scattered;
      } else {
        return color(0.0f, 0.0f, 0.0f);
      }
    } else {
      vec3 unit_direction = unit_vector(r.direction());
      auto t = 0.5f * (unit_direction.y() + 1.0f);
      auto miss_color = (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
      return current_attenuation * miss_color;
    }
  }

  return color(0.0f, 0.0f, 0.0f);
}

__global__ void create_camera(camera * cam, float aspect_ratio) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    point3 lookfrom(13.0f, 2.0f, 3.0f);
    point3 lookat(0.0f, 0.0f, 0.0f);
    vec3 vup(0.0f, 1.0f, 0.0f);
    float vfov = 20.0f;
    float aperture = 0.1f;
    float dist_to_focus = 10.0f;

    new (cam) camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus);
  }
}


__global__ void create_random_scene(hittable_list * world, hittable_ptr_t * objects, curandState * local_rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    size_t objects_size = 0;
    material * ground_material = new lambertian(color(0.5f, 0.5f, 0.5f));
    objects[objects_size++] = new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material);
    
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float choose_mat = random_double(local_rand_state);
        point3 center(a + 0.9f * random_double(local_rand_state), 0.2f, b + 0.9f * random_double(local_rand_state));

        if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
          material * sphere_material;
          if (choose_mat < 0.8f) {
            // diffuse
            auto albedo = color::random(local_rand_state) * color::random(local_rand_state);
            sphere_material = new lambertian(albedo);
          } else if (choose_mat < 0.95f) {
            // metal
            auto albedo = color::random(0.5f, 1.0f, local_rand_state);
            auto fuzz = random_double(0.0f, 0.5f, local_rand_state);
            sphere_material = new metal(albedo, fuzz);
          } else {
            // glass
            sphere_material = new dielectric(1.5f);
          }

          objects[objects_size++] = new sphere(center, 0.2f, sphere_material);
        }
      }
    }

    material * material1 = new dielectric(1.5f);
    objects[objects_size++] = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, material1);

    material * material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
    objects[objects_size++] = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2);

    material * material3 = new metal(color(0.7f, 0.6f, 0.5f), 0.0f);
    objects[objects_size++] = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, material3);

    new (world) hittable_list(objects, objects_size);
  }
}

__global__ void render_init(unsigned long long seed, int max_x, int max_y, curandState * rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 * frame_buffer,
                       int max_x, int max_y, int ns,
                       camera * cam, hittable_ptr_t world, curandState *rand_state, int max_depth) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 pixel_color(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < ns; ++s) {
      auto u = (i + random_double(&local_rand_state)) / (max_x - 1);
      auto v = (j + random_double(&local_rand_state)) / (max_y - 1);
      ray r = cam->get_ray(u, v, &local_rand_state);
      pixel_color += ray_color(r, world, max_depth, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    normalize_by_sample(pixel_color, ns);
    gamma_correction(pixel_color);
    frame_buffer[pixel_index] = pixel_color;
}


int main() {

  // Image

  const float aspect_ratio = 3.0f / 2.0f;
  const int image_width = 300;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 500;
  const int max_depth = 50;

  const int thread_dim_x = 8;
  const int thread_dim_y = 8;

  std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << samples_per_pixel << " samples per pixel ";
  std::cerr << "in " << thread_dim_x << "x" << thread_dim_y << " blocks.\n";

  // Allocate Frame Buffer
  int num_pixels = image_width * image_height;
  vec3 * frame_buffer;
  size_t frame_buffer_size = num_pixels * sizeof(*frame_buffer);

  checkCudaErrors(cudaMallocManaged(&frame_buffer, frame_buffer_size));

  // Allocate Random State
  curandState * d_rand_state;
  checkCudaErrors(cudaMalloc(&d_rand_state, num_pixels * sizeof(*d_rand_state)));

  int block_dim_x = (image_width + thread_dim_x - 1) / thread_dim_x;
  int block_dim_y = (image_height + thread_dim_y - 1) / thread_dim_y;
  dim3 blocks(block_dim_x, block_dim_y);
  dim3 threads(thread_dim_x, thread_dim_y);
  unsigned long long seed = time(nullptr);

  render_init<<<blocks, threads>>>(seed, image_width, image_height, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Camera
  camera * cam;
  checkCudaErrors(cudaMalloc(&cam, num_pixels * sizeof(*cam)));
  create_camera<<<1, 1>>>(cam, aspect_ratio);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // World
  hittable_list * world;
  checkCudaErrors(cudaMalloc(&world, sizeof(*world)));

  hittable_ptr_t * objects;
  checkCudaErrors(cudaMalloc(&objects, 500 * sizeof(*objects)));

  create_random_scene<<<1, 1>>>(world, objects, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto start = std::chrono::high_resolution_clock::now();

  // Render

  render<<<blocks, threads>>>(frame_buffer, image_width, image_height, samples_per_pixel, cam, world, d_rand_state, max_depth);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();

  auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
  std::cerr << "Time cost: " << timer_in_ms.count() << "ms.\n";


  // Output Frame Buffer as Image

  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  for (int j = image_height - 1; j >= 0; j--) {
      for (int i = 0; i < image_width; i++) {
          size_t pixel_index = j * image_width + i;
          int ir = int(255.99f * frame_buffer[pixel_index].x());
          int ig = int(255.99f * frame_buffer[pixel_index].y());
          int ib = int(255.99f * frame_buffer[pixel_index].z());
          std::cout << ir << " " << ig << " " << ib << "\n";
      }
  }

  // Clean Up
  // checkCudaErrors(cudaDeviceSynchronize());
  // free_world<<<1, 1>>>(d_list, d_world, d_camera);
  // checkCudaErrors(cudaGetLastError());
  // checkCudaErrors(cudaFree(d_camera));
  // checkCudaErrors(cudaFree(d_world));
  // checkCudaErrors(cudaFree(d_list));
  // checkCudaErrors(cudaFree(d_rand_state));
  // checkCudaErrors(cudaFree(frame_buffer));

  // cudaDeviceReset();

  std::cerr << "\nDone.\n";
}