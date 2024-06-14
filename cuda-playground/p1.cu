#include <iostream>
#include <algorithm>
#include <cuda/std/array>
#include <cassert>

#include "utils.h"

template <typename T, size_t S>
using CudaArray = cuda::std::array<T, S>;

template <typename T, size_t S> 
__global__ 
void kernel(CudaArray<T, S> *x)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for(int i = index; i < S; i += stride)
        (*x)[i] += 10;
}

int main(void)
{
    const int n = 10;
    CudaArray<float, n> *x;
    gpuErrchk(cudaMallocManaged(&x, n*sizeof(CudaArray<float, n>)));
    new (x) CudaArray<float, n>();

    std::transform(x->begin(), x->end(), x->begin(), [](...) { return 1.0f; });
    std::for_each(x->begin(), x->end(), [](auto z) { std::cout << z << ","; });
    std::cout << "\n";

    kernel<<<1,1>>>(x);
    
    gpuErrchk(cudaDeviceSynchronize());

    std::for_each(x->begin(), x->end(), [](auto z) { std::cout << z << ","; });
    std::cout << "\n\n";
    x->~CudaArray<float, n>();
}