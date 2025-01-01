# Ray Tracing in One Weekend
https://raytracing.github.io/books/RayTracingInOneWeekend.html
![cover](./gallery/gpu/image23-4k.png)

# Quick Start for Windows Platform

## How to Configure
1. Start Menu -> x64 Native Tools Command Prompt for VS 2022
2. cd to the repo
3. `mkdir build`
4. `cd build`
5. `cmake ..`

## How to Build

#### Build GPU Version
`cmake --build . --target gpu_ray_tracer --config Release`

#### Build CPU Version
`cmake --build . --target cpu_ray_tracer --config Release`

#### Build CPU Multi-Threaded Version
`cmake --build . --target cpu_multi_threading_ray_tracer --config Release`
