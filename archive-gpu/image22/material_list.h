#ifndef MATERIAL_LIST_H
#define MATERIAL_LIST_H

#include "material.h"

class material_list {
public:
  material_ptr *materials;
  size_t materials_size;
  size_t materials_capacity;

  __device__ material_list()
      : materials(nullptr), materials_size(0), materials_capacity(0) {}

  __device__ void add(material_ptr mat) {
    if (!this->materials) {
      this->materials = new material_ptr[1];
      this->materials_capacity = 1;
    }

    if (this->materials_size >= this->materials_capacity) {
      size_t new_materials_capacity = this->materials_capacity * 2;
      material_ptr *new_materials = new material_ptr[new_materials_capacity];
      for (size_t i = 0; i < this->materials_size; ++i) {
        new_materials[i] = this->materials[i];
      }

      delete[] this->materials;
      this->materials = new_materials;
      this->materials_capacity = new_materials_capacity;
    }

    this->materials[this->materials_size++] = mat;
  }

  __device__ ~material_list() {
    if (this->materials) {
      for (size_t i = 0; i < this->materials_size; ++i) {
        delete this->materials[i];
      }

      delete[] this->materials;
      this->materials = nullptr;
    }
  }
};

#endif