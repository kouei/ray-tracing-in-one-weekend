#include <chrono>
#include <iostream>
#include <limits>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "cuda_utility.h"

constexpr const float TMAX = std::numeric_limits<float>::max();

__device__ vec3 ray_color(const ray & r, hittable_ptr_t world) {
    hit_record rec;
    if (world->hit(r, 0.0f, TMAX, rec)) {
        return 0.5f * (rec.normal + vec3(1.0f, 1.0f, 1.0f));
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
    }
}

__global__ void render(vec3 * frame_buffer, int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hittable_ptr_t * world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    frame_buffer[pixel_index] = ray_color(r, world[0]);
}

__global__ void create_world(hittable_ptr_t * d_list, hittable_ptr_t * d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
        d_list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
        d_world[0] = new hittable_list(d_list, 2);
    }
}

__global__ void free_world(hittable_ptr_t * d_list, hittable_ptr_t * d_world) {
    delete d_list[0];
    delete d_list[1];
    delete d_world[0];
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

    // allocate Frame Buffer
    checkCudaErrors(cudaMallocManaged(&frame_buffer, frame_buffer_size));

    // make our world of hitables
    hittable_ptr_t * d_list;
    checkCudaErrors(cudaMalloc(&d_list, 2 * sizeof(*d_list)));
    hittable_ptr_t * d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(*d_world)));
    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

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
                                vec3(0.0, 0.0, 0.0),
                                d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    auto end = std::chrono::high_resolution_clock::now();

    auto timer_in_ms = std::chrono::duration<double, std::milli>(end - start);
    std::cerr << "Time cost: " << timer_in_ms.count() << "ms.\n";

    // Output FB as Image
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

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(frame_buffer));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}