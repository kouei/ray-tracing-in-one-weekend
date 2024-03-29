cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(RayTracerInOneWeekend VERSION 1.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(cpu_ray_tracer src/cpu/main.cc)
add_executable(cpu_multi_threading_ray_tracer src/cpu-multi-threading/main.cc)
add_executable(gpu_ray_tracer src/gpu/main.cu)
add_executable(gpu_ray_tracer_debug src/gpu/main.cu)

set_target_properties(gpu_ray_tracer PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(gpu_ray_tracer_debug PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(gpu_ray_tracer_debug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--device-debug>)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")