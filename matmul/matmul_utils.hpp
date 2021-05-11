#ifndef MATMUL_UTILS
#define MATMUL_UTILS

#include<iostream>
#include<cassert>
#include<ctime>

/*
Utility function that initializes a matix of size r*c
If identity is set to true, then matrix is initialized to identity matrix
Else, the matrix is set to random initialization
*/
void init_matrix(int* matrix, int r, int c, int identity=false);

/*
Display a matrix of size r*c in a neat 2D format
*/
void display_matrix(int* matrix, int r, int c);

/*
Multiply matrices a (size m*n) and b (size n*p) and store the result in 
matrix c.
CPU implementation with time complexity O(n^3)
*/
void cpu_matmul(int* a, int* b, int* c, int m, int n, int p);
#endif