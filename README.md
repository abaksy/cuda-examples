# cuda-examples
A repository of examples coded in CUDA C++

## matmul-v1

A basic matrix multiplication implemented in CUDA C++, where both matrices are loaded entirely and GPU threads act on the respective blocks

## matmul-v2

A Tiled implementation of Matrix Multiplication in CUDA C++ that takes advantage of cache performance by allowing each GPU block to multiply one tile
of the matrix using its own local cache (which has higher speed)
