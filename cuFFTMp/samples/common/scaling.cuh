#include "../iterators/box_iterator.hpp"

__global__
void scaling_kernel(BoxIterator<cufftComplex> begin, BoxIterator<cufftComplex> end, int rank, int size, size_t nx, size_t ny, size_t nz) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    begin += tid;
    if(begin < end) {
        // begin.x(), begin.y() and begin.z() are the global 3D coordinate of the data pointed by the iterator
        // begin->x and begin->y are the real and imaginary part of the corresponding cufftComplex element
        if(tid < 10) {
            printf("GPU data (after first transform): global 3D index [%d %d %d], local index %d, rank %d is (%f,%f)\n", 
                (int)begin.x(), (int)begin.y(), (int)begin.z(), (int)begin.i(), rank, begin->x, begin->y);
        }
        *begin = {begin->x / (float)(nx * ny * nz), begin->y / (float)(nx * ny * nz)};
    }
};