all: jacobi2d

jacobi2d : jacobi2d.C
	mpicxx -O2 -o jacobi2d jacobi2d.C

clean:
	rm -f jacobi2d