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
void kernel(CudaArray<int, S> *x, CudaArray<int, S> *y, CudaArray<int, S> *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S) {
        (*out)[i] = (*x)[i] + (*y)[i];
    }
}

int spec(int x, int y) {
    return x + y;
}

int main(void) {
    const int size = 1000;
    const int threads = 256;
    const int blocks = ceil(size/(float)threads);

    std::random_device rnd_dev;
    std::default_random_engine rnd_eng(rnd_dev());
    std::uniform_int_distribution<int> uniform_dist(0, 1000);

    CudaArray<int, size> *x, *y, *out;
    gpuErrchk(cudaMallocManaged(&x, sizeof(CudaArray<int, size>)));
    gpuErrchk(cudaMallocManaged(&y, sizeof(CudaArray<int, size>)));
    gpuErrchk(cudaMallocManaged(&out, sizeof(CudaArray<int, size>)));
    new (x) CudaArray<int, size>();
    new (y) CudaArray<int, size>();
    new (out) CudaArray<int, size>();

    std::transform(x->begin(), x->end(), x->begin(), [&](...) { return uniform_dist(rnd_eng); });
    std::transform(y->begin(), y->end(), y->begin(), [&](...) { return uniform_dist(rnd_eng); });

    std::array<int, size> expected; 
    for (int i = 0; i < size; i++) {
        expected[i] = spec((*x)[i], (*y)[i]);
    }

    kernel<<<blocks, threads>>>(x, y, out);
    
    gpuErrchk(cudaDeviceSynchronize());

    bool check = true;

    for (int i = 0; i < size; i++) {
        check = check && (*out)[i] == expected[i];
    }

    std::cout << "PASS: " << check << "\n";

    x->~CudaArray<int, size>();
    y->~CudaArray<int, size>();
    out->~CudaArray<int, size>();
}