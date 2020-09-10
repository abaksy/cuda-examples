#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matmul.h"

/*
Matrix multiplication:

For each element of output matrix C, assign a thread to it
Each of these threads operates in parallel and writes the result to the corresponding element of C
*/

__global__ void matrixMul(int* dA, int* dB, int* dC, int n)
{
	/*
	__global__ identifies this function as one that must be executed on GPU

	Row and Col are the indices of the result matrix element being written to 
	by the current thread
	*/
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int temp = 0;

	if (row < n && col < n)
	{
		for (int k = 0; k < n; ++k)
			temp += dA[row * n + k] * dB[k * n + col];

		dC[row * n + col] = temp;
	}
}



int main()
{
	size_t n = 1 << 10; //1024
	size_t bytes = n * n * sizeof(int);

	//Allocating memory on host
	int* hA = (int*)malloc(bytes);
	int* hB = (int*)malloc(bytes);
	int* hC = (int*)malloc(bytes);

	//Device pointers

	int* dA, * dB, * dC;

	//Allocate device memory (ie. memory on GPU)

	cudaMalloc(&dA, bytes);
	cudaMalloc(&dB, bytes);
	cudaMalloc(&dC, bytes);

	//initialize matrices

	init_matrices(hA, hB, n);

	//Copy data from host to device

	cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

	//Block and thread parameters
	/*
	Total number of threads = 1024*1024

	Each block has 16 * 16 threads, ie. 256 threads

	grid_size = ceil(1024 / 16) = 64

	Therefore

	64*64 grid of blocks, each block having 16*16 threads
	*/

	int block_size = 16;
	int grid_size = (int)ceil(n / block_size);

	//Initializing dim3 objects

	dim3 grid(grid_size, grid_size);
	dim3 threads(block_size, block_size);

	/*
	<<<grid_dimensions, block_dimensions>>>
	grid_dimensions: number of blocks in grid
	block_dimensions: number of threads per block
	*/
	float time;
	cudaEvent_t start, stop;


	//Launch Kernel
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	matrixMul << <grid, threads >> > (dA, dB, dC, n);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time to multiply (size=1024):  %f s \n", time/1000);
	//Copy result from device to host

	cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

	//Verify result using brute force
	/*if (verify_result(hA, hB, hC, n) == 1)
		printf("Successfully completed!\n");
	else
		printf("Error!\n");*/

	//Free memory on device
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	//Free memory on host
	free(hA);
	free(hB);
	free(hC);

	return 0;
}