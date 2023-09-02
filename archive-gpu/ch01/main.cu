#define CUDA_RAY_TRACER

#include <chrono>
#include <iostream>
#include "cuda_utility.h"

__global__ void render(float * frame_buffer, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if((i >= max_x) || (j >= max_y)) {
      return;
    }
    
    int pixel_index = j * max_x * 3 + i * 3;
    frame_buffer[pixel_index + 0] = float(i) / max_x;
    frame_buffer[pixel_index + 1] = float(j) / max_y;
    frame_buffer[pixel_index + 2] = 0.2;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    
    // Allocate Frame Buffer
    float * framebuffer;
    size_t frame_buffer_size = 3 * num_pixels * sizeof(*framebuffer);

    checkCudaErrors(cudaMallocManaged(&framebuffer, frame_buffer_size));

    auto start = std::chrono::high_resolution_clock::now();

    // Render our buffer
    int block_dim_x = (nx + tx - 1)/ tx;
    int block_dim_y = (ny + ty - 1) / ty;
    dim3 blocks(block_dim_x, block_dim_y);
    dim3 threads(tx, ty);

    render<<<blocks, threads>>>(framebuffer, nx, ny);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
    std::cerr << "Time cost: " << timer_in_ms.count() << "ms.\n";

    // Output FrameBuffer as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * 3 * nx + i * 3;
            float r = framebuffer[pixel_index + 0];
            float g = framebuffer[pixel_index + 1];
            float b = framebuffer[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(framebuffer));
}