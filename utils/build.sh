#!/bin/bash
mkdir target -p
clang++ src/*.cc -o target/main -O2 -Wall