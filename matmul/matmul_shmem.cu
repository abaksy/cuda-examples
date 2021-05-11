/*
Matrix multiplication in CUDA C++
using shared memory (shared between all threads in a single block)

Matrix sizes are M*N and P*Q
*/
#include"matmul_utils.hpp"

#define M 3
#define N 3
#define P 3
#define Q 3

__global__ void matmulKernel(int* a, int* b, int* c)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    __shared__ int sm[N];

    int i;
    int k = threadIdx.x;
    c[Q*x + y] = 0;

    sm[k] = a[N*x + k] * b[Q*k + y];

    __syncthreads();  //Wait for all threads in the block to finish, so that the sm array is populated fully

    for(i=0; i<N; ++i)
        c[Q*x + y] += sm[i];
}

int main()
{
    srand(time(0));
    int* a = new int[M * N * sizeof(int)];
    int* b = new int[P * Q * sizeof(int)];
    int* c = new int[M * Q * sizeof(int)];
    init_matrix(a, M, N);
    init_matrix(b, P, Q);

    std::cout<<"A =\n";
    display_matrix(a, M, N);
    std::cout<<"B =\n";
    display_matrix(b, P, Q);

    assert(N==P); //Necessary condition for matrix multiplication

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, M * N * sizeof(int));
    cudaMalloc((void**)&d_b, P * Q * sizeof(int));
    cudaMalloc((void**)&d_c, M * Q * sizeof(int));

    cudaMemcpy(d_a, a, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, P * Q * sizeof(int), cudaMemcpyHostToDevice);
    
    /*
    Each block of the grid takes care of multiplying one row of A with one column of B

    Each thread in the block takes care of one element multiplication between row of A and column of B
    */
    dim3 gridSize(M, Q);

    matmulKernel<<<gridSize, N>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, M * Q * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout<<"A*B =\n";
    display_matrix(c, M, Q);

    delete a;
    delete b;
    delete c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}