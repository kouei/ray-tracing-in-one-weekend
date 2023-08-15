cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(RayTracerInOneWeekend VERSION 1.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(cpu_ray_tracer src/cpu/main.cc)
add_executable(gpu_ray_tracer src/gpu/main.cu)

set_target_properties(gpu_ray_tracer PROPERTIES CUDA_ARCHITECTURES native)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")