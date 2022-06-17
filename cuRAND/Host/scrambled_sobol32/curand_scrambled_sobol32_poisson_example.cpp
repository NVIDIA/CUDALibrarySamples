/*
 * This program uses the host CURAND API to generate 100
 * quasirandom floats.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>

#include "curand_utils.h"

using data_type = unsigned int;

void run_on_device(const int &n, const unsigned long long &offset,
                   const unsigned int &num_dimensions, const double &lambda,
                   const curandOrdering_t &order, const curandRngType_t &rng,
                   const cudaStream_t &stream, curandGenerator_t &gen,
                   std::vector<data_type> &h_data) {

  data_type *d_data = nullptr;

  /* C data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data),
                        sizeof(data_type) * h_data.size()));

  /* Create quasi-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64));

  /* Set cuRAND to stream */
  CURAND_CHECK(curandSetStream(gen, stream));

  /* Set offset */
  CURAND_CHECK(curandSetGeneratorOffset(gen, offset));

  /* Set Dimension */
  CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(gen, num_dimensions));

  /* Set ordering */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Generate n floats on device */
  CURAND_CHECK(curandGeneratePoisson(gen, d_data, h_data.size(), lambda));

  /* Copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(h_data.data(), d_data,
                             sizeof(data_type) * h_data.size(),
                             cudaMemcpyDeviceToHost, stream));

  /* Sync stream */
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* Cleanup */
  CUDA_CHECK(cudaFree(d_data));
  CURAND_CHECK(curandDestroyGenerator(gen));
}

void run_on_host(const int &n, const unsigned long long &offset,
                 const unsigned int &num_dimensions, const double &lambda,
                 const curandOrdering_t &order, const curandRngType_t &rng,
                 const cudaStream_t &stream, curandGenerator_t &gen,
                 std::vector<data_type> &h_data) {

  /* Create quasi-random number generator */
  CURAND_CHECK(
      curandCreateGeneratorHost(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64));

  /* Set cuRAND to stream */
  CURAND_CHECK(curandSetStream(gen, stream));

  /* Set offset */
  CURAND_CHECK(curandSetGeneratorOffset(gen, offset));

  /* Set Dimension */
  CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(gen, num_dimensions));

  /* Set ordering */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Generate n floats on device */
  CURAND_CHECK(
      curandGeneratePoisson(gen, h_data.data(), h_data.size(), lambda));

  /* Cleanup */
  CURAND_CHECK(curandDestroyGenerator(gen));
}

int main(int argc, char *argv[]) {

  cudaStream_t stream = NULL;
  curandGenerator_t gen = NULL;
  curandRngType_t rng = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
  curandOrdering_t order = CURAND_ORDERING_QUASI_DEFAULT;

  const int n = 10;
  const double lambda = 10.0;

  const unsigned long long offset = 0ULL;
  const unsigned int num_dimensions = 1;

  /* Create stream */
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  /* Allocate n floats on host */
  std::vector<data_type> h_data(n, 0);
  run_on_host(n, offset, num_dimensions, lambda, order, rng, stream, gen,
              h_data);

  printf("Host\n");
  print_vector(h_data);
  printf("=====\n");

  run_on_device(n, offset, num_dimensions, lambda, order, rng, stream, gen,
                h_data);

  printf("Device\n");
  print_vector(h_data);
  printf("=====\n");

  /* Cleanup */
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}
