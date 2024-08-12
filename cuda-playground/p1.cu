#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>
#include <array>

#include <cuda/std/array>

#include "utils.h"

template <typename T, size_t S>
using CudaArray = cuda::std::array<T, S>;

template <typename T, size_t S> 
__global__ 
void kernel(CudaArray<T, S> *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S) {
        (*x)[i] += 10;
    }
}

template <size_t S>
std::array<int, S> spec(std::array<int, S> v) {
    std::transform(v.begin(), v.end(), v.begin(), [](int x) { return x + 10; });
    return v;
}

int main(void) {
    const int size = 1000;
    const int threads = 256;
    const int blocks = ceil(size/(float)threads);

    std::random_device rnd_dev;
    std::default_random_engine rnd_eng(rnd_dev());
    std::uniform_int_distribution<int> uniform_dist(0, 1000);

    std::array<int, size> inputs;
    std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](...) { return uniform_dist(rnd_eng); });

    std::array<int, size> expected = spec(inputs);

    CudaArray<float, size> *x;
    gpuErrchk(cudaMallocManaged(&x, size*sizeof(CudaArray<float, size>)));
    new (x) CudaArray<float, size>();
    std::copy(inputs.cbegin(), inputs.cend(), x->begin());

    kernel<<<blocks, threads>>>(x);
    
    gpuErrchk(cudaDeviceSynchronize());

    bool check = true;

    for (int i = 0; i < size; i++) {
        check = check && (int)((*x)[i]) == expected[i];
    }

    std::cout << "PASS: " << check << "\n";

    x->~CudaArray<float, size>();
}