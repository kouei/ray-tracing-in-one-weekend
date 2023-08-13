cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(RayTracerInOneWeekend VERSION 1.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_executable(RayTracerInOneWeekend src/cpu/main.cc)