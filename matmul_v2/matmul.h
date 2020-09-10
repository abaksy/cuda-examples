#ifndef MATMUL_H
#define MATMUL_H

#define SHMEM_SIZE 16*16*4

void init_matrices(int* hA, int* hB, size_t n);

int verify_result(int* hA, int* hB, int* hC, size_t n);

#endif