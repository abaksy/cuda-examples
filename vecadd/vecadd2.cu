/*
CUDA C++ Program to add 2 1-dimensional vectors of length N
1 blocks, N threads in block
*/
#include<iostream>
#include<cuda_runtime.h>
#define N 10

__global__ void vecadd(int* a, int* b, int* c)
{
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx]; 
}

int main()
{
    /* Set up variables on host*/
    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    /* Input values on host*/
    unsigned int size = N*sizeof(int);
    for(int i=0; i<N; ++i)
    {
        a[i] = 2*i;
        b[i] = 3*i+1;
    }

    /* Setting up variables on device. i.e. GPU */
    int* d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    /* Copy data from host to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    /* 
    Kernel Launch
    Grid contains 1 block 
    That block has N threads
    Hence index of vector is thread index
    */
    vecadd<<<1, N>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    /* Copy result from GPU device to host */
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    /* Print result */
    for(int i=0; i<N; ++i)
    {
        std::cout<<c[i]<<' ';
    }
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