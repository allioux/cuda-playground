#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>

#include <cuda/std/array>

#include "utils.h"

template <typename T, size_t S>
using CudaArray = cuda::std::array<T, S>;

template <size_t S> 
__global__ 
void kernel(CudaArray<int, S> *x, CudaArray<int, S> *out, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        int k = j * width + i; 
        (*out)[k] = (*x)[k] + 10;
    }
}

int spec(int x) {
    return x + 10;
}

int main(void) {
    const int size = 1000;
    const int square_size = size * size;
    dim3 dimGrid(ceil(size/(float)32), ceil(size/(float)32)); 
    dim3 dimBlock(32, 32);

    std::random_device rnd_dev;
    std::default_random_engine rnd_eng(rnd_dev());
    std::uniform_int_distribution<int> uniform_dist(0, 1000);

    CudaArray<int, square_size> *x, *out;
    gpuErrchk(cudaMallocManaged(&x, sizeof(CudaArray<int, square_size>)));
    gpuErrchk(cudaMallocManaged(&out, sizeof(CudaArray<int, square_size>)));
    new (x) CudaArray<int, square_size>();
    new (out) CudaArray<int, square_size>();

    std::transform(x->begin(), x->end(), x->begin(), [&](...) { return uniform_dist(rnd_eng); });

    std::array<int, square_size> expected; 
    for (int i = 0; i < square_size; i++) {
        expected[i] = spec((*x)[i]);
    }

    kernel<<<dimGrid, dimBlock>>>(x, out, size, size);
    
    gpuErrchk(cudaDeviceSynchronize());

    bool check = true;

    for (int i = 0; i < square_size; i++) {
        check = check && (*out)[i] == expected[i];
    }

    std::cout << "PASS: " << check << "\n";

    x->~CudaArray<int, square_size>();
    out->~CudaArray<int, square_size>();
}