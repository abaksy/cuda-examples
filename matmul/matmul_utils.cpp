#include"matmul_utils.hpp"

void init_matrix(int* matrix, int r, int c, int identity)
{
    if(identity)
    {
        assert(r==c);  //Identity matrix must be square matrix
        for(int i=0; i<r; ++i)
        {
            for(int j=0; j<c; ++j)
            {
                if(i==j)
                    matrix[i*c+j] = 1;
                else
                    matrix[i*c+j] = 0;
            }
        }
    }
    else
    {
        for(int i=0; i<r; ++i)
        {
            for(int j=0; j<c; ++j)
            {
                matrix[i*c+j] = rand()%10;
            }
        }
    }
}

void display_matrix(int* matrix, int r, int c)
{
    for(int i=0; i<r; ++i)
    {
        for(int j=0; j<c; ++j)
        {
            std::cout<<matrix[i*c+j]<<' ';
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}

void cpu_matmul(int* a, int* b, int* c, int m, int n, int p)
{
    for(int i=0; i<m; ++i)
    {
        for(int j=0; j<p; ++j)
        {
            c[i*p + j] = 0;
            for(int k=0; k<n; ++k)
            {
                c[i*p + j] += a[i*n + k] * b[k*p + j];
            }
        }
    }
}