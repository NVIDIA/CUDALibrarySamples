#ifndef __CUFFTMP_BOX_ITERATOR_HPP__
#define __CUFFTMP_BOX_ITERATOR_HPP__

#include <iterator>
#include <cstddef> 
#include <cufftXt.h>
#include <tuple>

/**
 * This iterator lets one iterate through the underlying data
 * associated to a cufftBox3d, and exposes the mapping
 * between global 3D coordinates (x, y, z) and local linear
 * indices.
 * 
 * This iterator can be used in __host__ or __device__ code
 */
template<typename T>
struct BoxIterator 
{
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;

    __host__ __device__ __forceinline__
    BoxIterator(size_t i, cufftBox3d box, T* ptr) : i_(i), box_(box), ptr_(ptr), 
                                                    lx_(box.upper[0] - box.lower[0]),
                                                    ly_(box.upper[1] - box.lower[1]),
                                                    lz_(box.upper[2] - box.lower[2]) {
        linear_to_box3d(i_, &x_, &y_, &z_);
    };

    __host__ __device__ __forceinline__
    BoxIterator& operator++() { increment(1); return *this; } 

    __host__ __device__
    BoxIterator operator++(int) { 
        BoxIterator tmp = *this; 
        ++(*this); 
        return tmp; 
    } 

    __host__ __device__ __forceinline__
    BoxIterator& operator--() { increment(-1); return *this; } 

    __host__ __device__
    BoxIterator operator--(int) { 
        BoxIterator tmp = *this; 
        --(*this); 
        return tmp; 
    }  
    
    __host__ __device__ __forceinline__
    BoxIterator& operator+=(difference_type rhs) { increment(rhs); return *this; }

    __host__ __device__ __forceinline__
    BoxIterator& operator-=(difference_type rhs) { increment(-rhs); return *this; }


    __host__ __device__ __forceinline__
    reference operator*() const { return ptr_[i()]; }

    __host__ __device__ __forceinline__
    pointer operator->() { return ptr_ + i(); }

    __host__ __device__ __forceinline__
    BoxIterator& operator[](difference_type rhs) const { return (*this + rhs); }

    __host__ __device__ __forceinline__ 
    friend difference_type operator-(const BoxIterator& a, const BoxIterator& b) {return a.i_ - b.i_; }

    __host__ __device__ __forceinline__ 
    friend BoxIterator operator-(const BoxIterator& a, difference_type n) { a -= n; return a; }

    __host__ __device__ __forceinline__ 
    friend BoxIterator operator+(const BoxIterator& a, difference_type n) { a += n; return a; }

    __host__ __device__ __forceinline__ 
    friend BoxIterator operator+(difference_type n, const BoxIterator& a) { return a+n; }

    __host__ __device__ __forceinline__ 
    friend bool operator==(const BoxIterator& a, const BoxIterator& b) { return a.i_ == b.i_; }

    __host__ __device__ __forceinline__ 
    friend bool operator!=(const BoxIterator& a, const BoxIterator& b) { return a.i_ != b.i_; }

    __host__ __device__ __forceinline__ 
    friend bool operator>(const BoxIterator& a, const BoxIterator& b) { return a.i_ > b.i_; }

    __host__ __device__ __forceinline__ 
    friend bool operator<(const BoxIterator& a, const BoxIterator& b) { return a.i_ < b.i_; }

    __host__ __device__ __forceinline__ 
    friend bool operator>=(const BoxIterator& a, const BoxIterator& b) { return a.i_ >= b.i_; }

    __host__ __device__ __forceinline__ 
    friend bool operator<=(const BoxIterator& a, const BoxIterator& b) { return a.i_ <= b.i_; }

    /**
     * Return the global X coordinate of the iterator
     */
    __host__ __device__ __forceinline__
    size_t x() const { return x_; }

    /**
     * Return the global Y coordinate of the iterator
     */
    __host__ __device__ __forceinline__
    size_t y() const { return y_; }

    /**
     * Return the global Z coordinate of the iterator
     */
    __host__ __device__ __forceinline__
    size_t z() const { return z_; }

    /**
     * Return the linear position of the iterator
     * in the local data buffer
     */
    __host__ __device__ __forceinline__
    size_t i() const {
        return (x_ - box_.lower[0]) * box_.strides[0] + (y_ - box_.lower[1]) * box_.strides[1] + (z_ - box_.lower[2]) * box_.strides[2]; 
    }

private:

    // Current 3D global index in the box
    size_t x_, y_, z_;
    // Current linear 3D index (not the location in memory)
    size_t i_;
    // Global box lower and upper corner and local strides
    const cufftBox3d box_;
    // Underlying data pointer
    T* ptr_;
    // Length of the X, Y and Z dimensions
    const size_t lx_, ly_, lz_;

    // Linear to 3D coordinates
    __host__ __device__ __forceinline__
    void linear_to_box3d(size_t i, size_t* x, size_t* y, size_t* z) {
        if(lx_ * ly_ * lz_ > 0) {
            *x  =   i  / (ly_ * lz_);
            i  -= (*x) * (ly_ * lz_);
            *y  =   i  / (lz_);
            i  -= (*y) * (lz_);
            *z  =   i;
        } else {
            *x = 0;
            *y = 0;
            *z = 0;
        }
        *x += box_.lower[0];
        *y += box_.lower[1];
        *z += box_.lower[2];
    }

    // Increment/decrement by n
    __host__ __device__ __forceinline__
    void increment(difference_type n) {
        i_ += n;
        linear_to_box3d(i_, &x_, &y_, &z_);
    }

};

template<typename T> __host__ __device__ __forceinline__ 
BoxIterator<T> BoxIteratorBegin(cufftBox3d box, T* ptr) {
    return BoxIterator<T>(0, box, ptr);
};

template<typename T> __host__ __device__ __forceinline__
BoxIterator<T> BoxIteratorEnd(cufftBox3d box, T* ptr) {
    return BoxIterator<T>( (box.upper[0] - box.lower[0]) * (box.upper[1] - box.lower[1]) * (box.upper[2] - box.lower[2]), box, ptr);
};

template<typename T>
std::pair<BoxIterator<T>,BoxIterator<T>> BoxIterators(cufftBox3d box, T* ptr) {
    return {BoxIteratorBegin<T>(box, ptr),BoxIteratorEnd<T>(box, ptr)};
}

size_t slabs_displacement(size_t length, int rank, int size) {
    int ranks_cutoff = length % size;
    return (rank < ranks_cutoff ? rank * (length / size + 1) : ranks_cutoff * (length / size + 1) + (rank - ranks_cutoff) * (length / size));
}

cufftBox3d buildCufftBox3d(cufftXtSubFormat format, cufftType type, int rank, int size, size_t nx, size_t ny, size_t nz) {
    if(format == CUFFT_XT_FORMAT_INPLACE) {
        size_t x_start      = slabs_displacement(nx, rank,   size);
        size_t x_end        = slabs_displacement(nx, rank+1, size);
        size_t my_ny        = ny;
        size_t my_nz        = nz;
        size_t my_nz_padded = (type == CUFFT_C2C || type == CUFFT_Z2Z) ? my_nz : 2*(nz/2 + 1);
        return {
            {x_start, 0, 0}, {x_end, my_ny, my_nz}, {my_ny * my_nz_padded, my_nz_padded, 1}
        };
    } else {
        size_t y_start      = slabs_displacement(ny, rank,   size);
        size_t y_end        = slabs_displacement(ny, rank+1, size);
        size_t my_nx        = nx;
        size_t my_nz        = (type == CUFFT_C2C || type == CUFFT_Z2Z) ? nz : (nz/2 + 1);
        size_t my_nz_padded = my_nz;
        return {
            {0, y_start, 0}, {my_nx, y_end, my_nz}, {(y_end-y_start) * my_nz_padded, my_nz_padded, 1}
        };
    }
}

template<typename T>
BoxIterator<T> BoxIteratorBegin(cufftXtSubFormat format, cufftType type, int rank, int size, size_t nx, size_t ny, size_t nz, T* ptr) {
    cufftBox3d box = buildCufftBox3d(format, type, rank, size, nx, ny, nz);
    return BoxIteratorBegin<T>(box, ptr);
}

template<typename T>
BoxIterator<T> BoxIteratorEnd(cufftXtSubFormat format, cufftType type, int rank, int size, size_t nx, size_t ny, size_t nz, T* ptr) {
    cufftBox3d box = buildCufftBox3d(format, type, rank, size, nx, ny, nz);
    return BoxIteratorEnd<T>(box, ptr);
}

template<typename T>
std::pair<BoxIterator<T>,BoxIterator<T>> BoxIterators(cufftXtSubFormat format, cufftType type, int rank, int size, size_t nx, size_t ny, size_t nz, T* ptr) {
    return {BoxIteratorBegin<T>(format, type, rank, size, nx, ny, nz, ptr),BoxIteratorEnd<T>(format, type, rank, size, nx, ny, nz, ptr)};
}

#endif // __CUFFTMP_BOX_ITERATOR_HPP__