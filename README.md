# cuda-examples
A repository of examples coded in CUDA C++
All examples were compiled using NVCC version 10.1 on Linux v 5.41

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

