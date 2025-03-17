all: jacobi2d

jacobi2d : jacobi2d.cpp
	mpicxx -O2 -o jacobi2d jacobi2d.cpp

clean:
	rm -f jacobi2d
