#!/bin/bash
mkdir bin -p
clang++ src/*.cc -o bin/main -std=c++17 -O2 -Wno-unused-result -Wshadow -Wall