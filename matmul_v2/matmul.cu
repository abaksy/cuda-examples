#include"matmul.h"
#include<stdlib.h>

void init_matrices(int* hA, int* hB, size_t n)
{
	/*
	Initialize matrices A and B with random values
	*/
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			hA[i * n + j] = rand() % 100;
			hB[i * n + j] = rand() % 100;
		}
	}
}

int verify_result(int* hA, int* hB, int* hC, size_t n)
{
	/*
	Verify result calculated on GPU against result calculated by Brute Force method on CPU
	*/
	int* actual_C = (int*)malloc(n * n * sizeof(int));
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			actual_C[i * n + j] = 0;
			for (size_t k = 0; k < n; ++k)
			{
				actual_C[i * n + j] += hA[i * n + k] * hB[k * n + j];
			}
		}
	}

	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			if (actual_C[i * n + j] != hC[i * n + j])
				return 0;
		}
	}
	return 1;
}