#include <chrono>
#include <iostream>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "cuda_utility.h"
#include "color.h"

constexpr const float TMAX = std::numeric_limits<float>::max();

__device__ vec3 ray_color(const ray & r, hittable_ptr_t world, curandState * local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if (world->hit(cur_ray, 0.001f, TMAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0f, 0.0f, 0.0f);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0f, 0.0f, 0.0f); // exceeded max recursion times
}

__global__ void render_init(int max_x, int max_y, curandState * rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 * frame_buffer,
                       int max_x, int max_y, int ns,
                       camera_ptr_t * cam, hittable_ptr_t * world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) {
        return;
    }

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 pixel_color(0.0f, 0.0f, 0.0f);
    for(int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = cam[0]->get_ray(u,v);
        pixel_color += ray_color(r, world[0], &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    normalize_by_sample(pixel_color, ns);
    gamma_correction(pixel_color);
    frame_buffer[pixel_index] = pixel_color;
}

__global__ void create_world(hittable_ptr_t * d_list, hittable_ptr_t * d_world, camera_ptr_t * d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0.0f, 0.0f,-1.0f), 0.5f, new lambertian(vec3(0.8f, 0.3f, 0.3f)));
        d_list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
        d_list[2] = new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f, new metal(vec3(0.8f, 0.6f, 0.2f), 1.0f));
        d_list[3] = new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, new metal(vec3(0.8f, 0.8f, 0.8f), 0.3f));
        d_world[0]  = new hittable_list(d_list, 4);
        d_camera[0] = new camera();
    }
}

__global__ void free_world(hittable_ptr_t * d_list, hittable_ptr_t * d_world, camera_ptr_t * d_camera) {
    for(int i = 0; i < 4; i++) {
        delete (reinterpret_cast<sphere *>(d_list[i]))->mat_ptr;
        delete d_list[i];
    }

    delete d_world[0];
    delete d_camera[0];
}

int main() {
    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    vec3 * frame_buffer;
    size_t frame_buffer_size = num_pixels * sizeof(*frame_buffer);

    // Allocate Frame Buffer
    checkCudaErrors(cudaMallocManaged(&frame_buffer, frame_buffer_size));

    // Allocate Random State
    curandState * d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, num_pixels * sizeof(*d_rand_state)));

    // make our world of hitables & the camera
    hittable_ptr_t * d_list;
    checkCudaErrors(cudaMalloc(&d_list, 4 * sizeof(*d_list)));
    hittable_ptr_t * d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(*d_world)));
    camera_ptr_t * d_camera;
    checkCudaErrors(cudaMalloc(&d_camera, sizeof(*d_camera)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    // Render our buffer

    int block_dim_x = (nx + tx - 1) / tx;
    int block_dim_y = (ny + ty - 1) / ty;
    dim3 blocks(block_dim_x, block_dim_y);
    dim3 threads(tx, ty);
    
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(frame_buffer, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    auto timer_in_ms = std::chrono::duration<float, std::milli>(end - start);
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
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(frame_buffer));

    cudaDeviceReset();
}