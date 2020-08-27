image.ppm: main
	./main > image.ppm

main: main.cc #vec3.h color.h ray.h
	clang++ main.cc -o main -O2 -Wall

.PHONY: clean

clean:
	rm image.ppm main