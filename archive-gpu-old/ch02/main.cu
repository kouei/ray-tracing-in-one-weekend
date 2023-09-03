#include <chrono>
#include <iostream>
#include "vec3.h"
#include "cuda_utility.h"

__global__ void render(vec3 * frame_buffer, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = j * max_x + i;
    frame_buffer[pixel_index] = vec3( float(i) / max_x, float(j) / max_y, 0.2f);
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    vec3 * frame_buffer;
    size_t frame_buffer_size = num_pixels * sizeof(*frame_buffer);

    // Allocate Frame Buffer
    checkCudaErrors(cudaMallocManaged(&frame_buffer, frame_buffer_size));

    auto start = std::chrono::high_resolution_clock::now();
    
    // Render our buffer
    int block_dim_x = (nx + tx - 1)/ tx;
    int block_dim_y = (ny + ty - 1) / ty;
    dim3 blocks(block_dim_x, block_dim_y);
    dim3 threads(tx, ty);

    render<<<blocks, threads>>>(frame_buffer, nx, ny);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
    std::cerr << "Time cost: " << timer_in_ms.count() << "ms.\n";

    // Output FrameBuffer as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * frame_buffer[pixel_index].r());
            int ig = int(255.99 * frame_buffer[pixel_index].g());
            int ib = int(255.99 * frame_buffer[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(frame_buffer));
}