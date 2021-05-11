/*
Program to add 2 matrics of size M * N in CUDA C++
Using 2-D grid of M*N size (i.e. grid contains M*N blocks arranged in 2D fashion)
Each block contains 1 thread
*/
#include<iostream>
#include<cuda_runtime.h>
#define M 7
#define N 3

__global__ void matAdd(int* a, int* b, int* c)
{
    int idx = blockIdx.x * gridDim.y + blockIdx.y;
    c[idx] = a[idx] + b[idx];
}

__host__ void print_matrix(int* matrix)
{
    for(int i=0; i<M; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            std::cout<<matrix[i*N+j]<<' ';
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}

int main()
{
    int size = M * N * sizeof(int);
    int* a = new int[size];
    int* b = new int[size];
    int* c = new int[size];

    for(int i=0; i<M; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            a[i*N + j] = i; //Fill your own values here
            b[i*N + j] = j; //Fill your own values here
        }
    }
    std::cout<<"MATRIX A =\n";
    print_matrix(a);
    std::cout<<"MATRIX B =\n";
    print_matrix(b);

    /* Setting up variables on device. i.e. GPU */
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    /* Copy data from host to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    /* 
    Kernel Launch
    Grid contains M*N blocks
    Each block has 1 thread
    Hence index of matrix element is
    blockIdx.x* gridSize.y + blockIdx.y
    */
    dim3 gridSize(M, N);
    matAdd<<<gridSize, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    /* Copy result from GPU device to host */
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    /* Print result */
    std::cout<<"A + B =\n";
    print_matrix(c);
    std::cout<<'\n';

    /* Cleanup device and host memory */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete a;
    delete b;
    delete c;

    return 0;
}
