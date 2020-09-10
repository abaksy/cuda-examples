#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "matmul.h"

/*
Tiled matrix multiplication

At each time, the multiplication handles only one tile of A and B at a time

Size of each tile is size of one thread block, ie. 16threads * 16 threads
*/

__global__ void tiledMatrixMul(int* a, int* b, int* c, size_t n, int tile_size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Statically allocated shared memory
	__shared__ int s_a[SHMEM_SIZE];
	__shared__ int s_b[SHMEM_SIZE];

	// Accumulate in temporary variable
	int temp_sum = 0;

	// Sweep tile across matrix
	for (int i = 0; i < n; i += blockDim.x)
	{
		// Load in elements for this tile
		s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * n + i + threadIdx.x];
		s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * n + threadIdx.y * n + col];

		// Wait for both tiles to be loaded in before doing computation
		__syncthreads();

		// Do matrix multiplication on the small matrix
		for (int j = 0; j < blockDim.x; j++) 
		{
			temp_sum +=s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
		}

		// Wait for all threads to finish using current tiles before loading in new
		// ones
		__syncthreads();
	}

	// Write back results
	c[row * n + col] = temp_sum;
}

int main()
{
	int n = 1 << 10; //1024

	size_t size = n * n * sizeof(int);
	
	//Pointers to host memory
	int* hA, * hB, * hC;

	//Pointer to device memory
	int* dA, * dB, * dC;

	//Allocate host memory
	hA = (int*)malloc(size);
	hB = (int*)malloc(size);
	hC = (int*)malloc(size);

	//Allocate device memory

	cudaMalloc(&dA, size);
	cudaMalloc(&dB, size);
	cudaMalloc(&dC, size);

	//Initialize matrices
	init_matrices(hA, hB, n);

	//Copy data from host to device

	cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
	
	//Threads per block
	int BLOCK_SIZE = 16;

	//BLocks in each dimension
	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	//Launch kernel
	tiledMatrixMul <<<grid, threads >>> (dA, dB, dC, n, BLOCK_SIZE);

	//Copy result from device
	cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

	//Verify result
	if (verify_result(hA, hB, hC, n) == 1)
		printf("Completed Successfully!\n");
	else
		printf("Error!\n");

	//Free host memory
	free(hA);
	free(hB);
	free(hC);

	//Free device memory
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return 0;

}