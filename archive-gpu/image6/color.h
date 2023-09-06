#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "vec3.h"
#include <cuda_runtime.h>
#include <iostream>

using color = vec3;

__host__ void write_color(std::ostream &out, color pixel_color,
                          int samples_per_pixel) {
  float r = pixel_color.x();
  float g = pixel_color.y();
  float b = pixel_color.z();

  // Divide the color by the number of samples.
  float scale = 1.0f / samples_per_pixel;
  r *= scale;
  g *= scale;
  b *= scale;

  // Write the translated [0,255] value of each color component.
  static const interval intensity(0.000f, 0.999f);
  out << static_cast<int>(256.0f * intensity.clamp(r)) << ' '
      << static_cast<int>(256.0f * intensity.clamp(g)) << ' '
      << static_cast<int>(256.0f * intensity.clamp(b)) << '\n';
}

#endif