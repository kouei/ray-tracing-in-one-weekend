#include "cuda_utility.h"
#include "vec3.h"
#include <chrono>
#include <iostream>

__global__ void render(vec3 * frame_buffer, int image_width, int image_height) {
  int image_x = threadIdx.x + blockIdx.x * blockDim.x;
  int image_y = threadIdx.y + blockIdx.y * blockDim.y;
  if (image_x >= image_width || image_y >= image_height) {
    return;
  }

  int pixel_index = image_y * image_width + image_x;

  float r = float(image_x) / (image_width - 1);
  float g = float(image_y) / (image_height - 1);
  float b = 0.0f;

  frame_buffer[pixel_index] = vec3(r, g, b);
}

int main() {

  // Image

  int image_width = 1920;
  int image_height = 1080;

  const int samples_per_pixel = 500;
  const int n_thread_x = 16;
  const int n_thread_y = 16;

  std::clog << "Image Size = " << image_width << "x" << image_height << "\n";
  std::clog << "Samples Per Pixel = " << samples_per_pixel << "\n";
  std::clog << "Block Dim (a x b threads) = " << n_thread_x << "x" << n_thread_y << "\n";

  // Allocate Frame Buffer
  int n_pixels = image_width * image_height;
  vec3 * frame_buffer;
  size_t frame_buffer_size = n_pixels * sizeof(*frame_buffer);

  checkCudaErrors(cudaMallocManaged(&frame_buffer, frame_buffer_size));

  // Render

  int n_block_x = (image_width + n_thread_x - 1) / n_thread_x;
  int n_block_y = (image_height + n_thread_y - 1) / n_thread_y;
  dim3 blocks(n_block_x, n_block_y);
  dim3 threads(n_thread_x, n_thread_y);

  auto start = std::chrono::high_resolution_clock::now();

  render<<<blocks, threads>>>(frame_buffer, image_width, image_height);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
  std::clog << "Time Cost = " << static_cast<int>(timer_in_ms.count() + 0.999f) << " ms\n";

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