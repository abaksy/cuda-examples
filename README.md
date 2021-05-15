# cuda-examples
A repository of examples coded in CUDA C++
All examples were compiled using NVCC version 10.1 on Linux v 5.4

## Setup on Linux

1) Install Nvidia drivers for the installed Nvidia GPU. On Ubuntu-based distributions this can be done from the software & updates app, <br>
in the tab listed as "Additional Drivers" (make sure to install the recommended version of Nvidia drivers)

2) After installing and restarting, verify that the drivers were installed by running 
```
nvidia-smi
```
in a terminal window. The output should list the name of the installed card, along with some usage statistics 

3) Install the ```nvcc``` compiler using the package manager
```
sudo apt install nvidia-cuda-toolkit
```

4) Verify the installation using
```
nvcc -V
```
or 
```
nvcc --version
```

## vecadd

CUDA implementation of vector-vector addition, adding vectors of length N, using:
* One thread per block, N blocks
* N threads in one block, grid contains only one block
* M threads in block, N/M blocks

## matadd

CUDA implementation of matrix-matrix addition, adding matrices of size M x N, using:
* One thread per block, 2D grid of MxN blocks
* MxN threads in one block, grid contains only one block

## matmul

CUDA implementation of matrix-matrix multiplication, with matrices of size MxN and PxQ (where N = Q)
Implementation using:
* 2D grid of size MxQ blocks, with 1 thread in each block 
* 2D grid of size MxQ blocks, with N threads in each block and shared memory (block level)

