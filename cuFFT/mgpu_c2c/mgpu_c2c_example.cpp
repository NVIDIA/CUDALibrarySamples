#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftXt.h>

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL
// *************** FOR ERROR CHECKING *******************

using cpudata_t = std::vector<std::complex<float>>;
using gpus_t    = std::vector<int>;
using dim_t     = std::array<size_t, 3>;

void fill_array( cpudata_t &array ) {
    std::mt19937                          gen( 3 );  // certified random number
    std::uniform_real_distribution<float> dis( 0.0f, 1.0f );

    for ( size_t i = 0; i < array.size( ); ++i ) {
        float real = dis( gen );
        float imag = dis( gen );
        array[i]   = { real, imag };
    };
};

/** Single GPU version of cuFFT plan for reference. */
void single( dim_t fft, cpudata_t &h_data_in, cpudata_t &h_data_out ) {

    cufftHandle plan {};

    CUFFT_CALL( cufftCreate( &plan ) );

    size_t workspace_size;
    CUFFT_CALL( cufftMakePlan3d( plan, fft[0], fft[1], fft[2], CUFFT_C2C, &workspace_size ) );

    void * d_data;
    size_t datasize = h_data_in.size( ) * sizeof( std::complex<float> );

    CUDA_RT_CALL( cudaMalloc( &d_data, datasize ) );
    CUDA_RT_CALL( cudaMemcpy( d_data, h_data_in.data( ), datasize, cudaMemcpyHostToDevice ) );

    CUFFT_CALL( cufftXtExec( plan, d_data, d_data, CUFFT_FORWARD ) );

    CUDA_RT_CALL( cudaMemcpy( h_data_out.data( ), d_data, datasize, cudaMemcpyDeviceToHost ) );
    CUDA_RT_CALL( cudaFree( d_data ) );

    CUFFT_CALL( cufftDestroy( plan ) );
};

/** Since cuFFT 10.4.0 cufftSetStream can be used to associate a stream with
 * multi-GPU plan. cufftXtExecDescriptor synchronizes efficiently to the stream
 * before and after execution. Please refer to
 * https://docs.nvidia.com/cuda/cufft/index.html#function-cufftsetstream for
 * more information.
 * cuFFT by default executes multi-GPU plans in synchronous manner.
 * */

void spmg( dim_t fft, gpus_t gpus, cpudata_t &h_data_in, cpudata_t &h_data_out, cufftXtSubFormat_t subformat ) {

    // Initiate cufft plan
    cufftHandle plan {};
    CUFFT_CALL( cufftCreate( &plan ) );

#if CUFFT_VERSION >= 10400
    // Create CUDA Stream
    cudaStream_t stream {};
    CUDA_RT_CALL( cudaStreamCreate( &stream ) );
    CUFFT_CALL( cufftSetStream( plan, stream ) );
#endif

    // Define which GPUS are to be used
    CUFFT_CALL( cufftXtSetGPUs( plan, gpus.size( ), gpus.data( ) ) );

    // Create the plan
    // With multiple gpus, worksize will contain multiple sizes
    size_t workspace_sizes[gpus.size( )];
    CUFFT_CALL( cufftMakePlan3d( plan, fft[0], fft[1], fft[2], CUFFT_C2C, workspace_sizes ) );

    cudaLibXtDesc *indesc;

    // Copy input data to GPUs
    CUFFT_CALL( cufftXtMalloc( plan, &indesc, subformat ) );
    CUFFT_CALL( cufftXtMemcpy( plan,
                               reinterpret_cast<void *>( indesc ),
                               reinterpret_cast<void *>( h_data_in.data( ) ),
                               CUFFT_COPY_HOST_TO_DEVICE ) );

    // Execute the plan
    CUFFT_CALL( cufftXtExecDescriptor( plan, indesc, indesc, CUFFT_FORWARD ) );

    // Copy output data to CPU
    CUFFT_CALL( cufftXtMemcpy( plan,
                               reinterpret_cast<void *>( h_data_out.data( ) ),
                               reinterpret_cast<void *>( indesc ),
                               CUFFT_COPY_DEVICE_TO_HOST ) );

    CUFFT_CALL( cufftXtFree( indesc ) );
    CUFFT_CALL( cufftDestroy( plan ) );

#if CUFFT_VERSION >= 10400
    CUDA_RT_CALL( cudaStreamDestroy( stream ) );
#endif
};

/** Runs single and multi-GPU version of cuFFT plan then compares results.
 * Maximum FFT size limited by single GPU memory.
 * */
int main( ) {

    dim_t fft = { 256, 256, 256 };
    // can be {0, 0} to run on single-GPU system or if GPUs are not of same architecture
    gpus_t gpus = { 0, 1 };

    size_t element_count = fft[0] * fft[1] * fft[2];

    cpudata_t data_in( element_count );
    fill_array( data_in );

    cpudata_t data_out_reference( element_count, { -1.0f, -1.0f } );
    cpudata_t data_out_test( element_count, { -0.5f, -0.5f } );

    cufftXtSubFormat_t decomposition = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;

    spmg( fft, gpus, data_in, data_out_test, decomposition );
    single( fft, data_in, data_out_reference );

    // The cuFFT library doesn't guarantee that single-GPU and multi-GPU cuFFT
    // plans will perform mathematical operations in same order. Small
    // numerical differences are possible.

    // verify results
    double error {};
    double ref {};
    for ( size_t i = 0; i < element_count; ++i ) {
        error += std::norm( data_out_test[i] - data_out_reference[i] );
        ref += std::norm( data_out_reference[i] );
    };

    double l2_error = ( ref == 0.0 ) ? std::sqrt( error ) : std::sqrt( error ) / std::sqrt( ref );
    if ( l2_error < 0.001 ) {
        std::cout << "PASSED with L2 error = " << l2_error << std::endl;
    } else {
        std::cout << "FAILED with L2 error = " << l2_error << std::endl;
    };
};
