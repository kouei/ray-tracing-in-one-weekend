@echo off
doskey buildcpu=mkdir .\bin $T CL src\cpu\*.cc /EHsc /Fo:bin\ /Fe:bin\main.exe /O2 /Wall /wd4711 /wd4710 /wd5045 /wd4514 /wd4820 /wd4100
doskey buildgpu=mkdir .\bin $T nvcc src\gpu\*.cu -o bin\main --gpu-architecture=native