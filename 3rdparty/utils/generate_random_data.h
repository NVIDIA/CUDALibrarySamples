#pragma once
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/gallery/random.h>

template <typename MatrixType>
MatrixType generate_random_sparse_matrix(int num_rows, int num_cols, int num_nonzeros)
{
    // int num_nonzeros = num_rows * num_cols * sparsity;
    MatrixType A(num_rows, num_cols, num_nonzeros);
    cusp::gallery::random(A, num_rows, num_cols, num_nonzeros);
    return A;
}

void generate_random_matrix(float *data, int len)
{
    for (int i = 0; i < len; i++)
    {
        data[i] = (rand() + 0.0f) / RAND_MAX;
    }
}