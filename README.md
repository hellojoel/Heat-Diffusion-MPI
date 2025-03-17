# Parallel Heat Diffusion Simulation with OpenMPI

## Overview
This project implements a parallel computing model to simulate heat diffusion using the **2D Jacobi iteration method** in **C++ with OpenMPI**. The simulation models heat distribution over a 2D grid, leveraging parallel processing to improve computational efficiency.

## Features
- **Parallelized using OpenMPI** for distributed memory systems.
- **2D Jacobi iteration method** to approximate steady-state heat distribution.
- **Customizable grid size and iteration count**.
- **Performance optimization using domain decomposition**.

## Usage
Run the program with a specified number of processes:
```sh
mpirun -np <# of processes> ./jacobi2d <input filename> <# of timesteps> <size_X> <size_Y>
```
Example:
```sh
mpirun -np 4 ./jacobi2d 64x64-heatmap.csv 100 64 64
```
This runs the simulation on a `64x64` grid for `100` iterations using 4 processes.

## Example
- Initial temperature distribution
- Final temperature distribution
- Speedup analysis

## Future Improvements
- **Parallel I/O calls** and **MPI Cartesian Topologies** for optimized efficiency.
- Support for **adaptive mesh refinement**.
- Extend to **3D heat diffusion simulations**.


