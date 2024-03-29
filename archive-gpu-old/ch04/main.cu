#include <chrono>
#include <iostream>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "cuda_utility.h"

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ vec3 ray_color(const ray& r) {
    if (hit_sphere(vec3(0, 0, -1), 0.5, r)) {
        return vec3(1, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 * frame_buffer, int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    frame_buffer[pixel_index] = ray_color(r);
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
    int block_dim_x = (nx + tx - 1) / tx;
    int block_dim_y = (ny + ty - 1) / ty;
    dim3 blocks(block_dim_x, block_dim_y);
    dim3 threads(tx, ty);

    render<<<blocks, threads>>>(frame_buffer, nx, ny,
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();

    auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
    std::cerr << "Time cost: " << timer_in_ms.count() << "ms.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j  *nx + i;
            int ir = int(255.99 * frame_buffer[pixel_index].r());
            int ig = int(255.99 * frame_buffer[pixel_index].g());
            int ib = int(255.99 * frame_buffer[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(frame_buffer));
}