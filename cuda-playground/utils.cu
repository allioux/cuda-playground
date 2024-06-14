#include <iostream>
#include "utils.h"

void gpuMemInfos() {
    size_t* free;
    size_t* total;
    gpuErrchk(cudaMallocManaged(&free, sizeof(size_t)));
    gpuErrchk(cudaMallocManaged(&total, sizeof(size_t)));
    gpuErrchk(cudaMemGetInfo(free, total));
    std::cout << "Free: " << *free << " | Total: " << *total << "\n";
}