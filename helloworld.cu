#include<iostream>

__global__ void myKernel(){

}

int main()
{
    myKernel<<<1,1>>>();
    std::cout<<"Hello World!\n";
    return 0;
}
