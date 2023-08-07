# Ray Tracing in One Weekend
https://raytracing.github.io/books/RayTracingInOneWeekend.html

# Quick Start for Windows Platform

## How to Build
1. Start Menu -> x64 Native Tools Command Prompt for VS 2022
2. cd to the repo
3. Run `init.bat`
4. Run `buildd` to build ray tracer

## How to Render Image
1. Start Menu -> Windows PowerShell
2. cd to the repo
3. Run `Import-Module -Force .\init.psm1`
4. Run `Render-Output -Filename <image-filename>`
5. The .ppm image will be generated in the **bin** folder