#include <cstdio>
#include <memory>
#include <mutex>
#include <tuple>
#include <sstream>

#include "kernel_helpers.h"
#include "kernels.h"

#include <cufftMp.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(ans) { cuda_check((ans), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        throw std::runtime_error("CUDA error");
    }
}

#define NVSHMEM_CHECK(ans) { nvshmem_check((ans), __FILE__, __LINE__); }
inline void nvshmem_check(int code, const char *file, int line)
{
    if (code != 0) {
        fprintf(stderr,"NVSHMEM_CHECK: %d %s %d\n", code, file, line);
        throw std::runtime_error("NVSHMEM error");
    }
}

#define CUFFT_CHECK(ans) { cufft_check((ans), __FILE__, __LINE__); }
inline void cufft_check(int code, const char *file, int line, bool abort=true)
{
    if (code != CUFFT_SUCCESS) {
        fprintf(stderr,"CUFFT_CHECK: %d %s %d\n", code, file, line);
        throw std::runtime_error("CUFFT error");
    }
}

namespace cufftmp_jax {

namespace {

/**
 * Used to cache a plan accross executions
 * Planning can take a long time, and should only be done
 * once whenever possible.
 */

struct plan_cache {

    const std::int64_t nx, ny, nz;
    const std::int64_t count;
    
    cufftHandle plan_inplace_to_shuffled;
    cufftHandle plan_shuffled_to_inplace;

    float2* inout_d;
    float2* scratch_d;

    plan_cache(std::int64_t nx, 
               std::int64_t ny, 
               std::int64_t nz, 
               std::int64_t count, 
               cufftHandle plan0, 
               cufftHandle plan1, 
               float2* inout_d,
               float2* scratch_d)
        : nx(nx), ny(ny), nz(nz), count(count), 
          plan_inplace_to_shuffled(plan0), 
          plan_shuffled_to_inplace(plan1),
          inout_d(inout_d), scratch_d(scratch_d) {};

    static std::unique_ptr<plan_cache> create(std::int64_t nx, std::int64_t ny, std::int64_t nz) {

        // Initialize NVSHMEM, do basic checks
        nvshmem_init();
        int num_pes = nvshmem_n_pes();
        if(nx % num_pes != 0 || ny % num_pes != 0) {
            std::stringstream sstr;
            sstr << "Invalid configuration; nx = " << nx << " and ny = " << ny << " need to be divisible by the number of PEs = " << num_pes << "\n";
            throw std::runtime_error(sstr.str());
        }

        // Create plan #1
        // This plan will be used for (Slabs_X + CUFFT_FORWARD or Slabs_Y + CUFFT_INVERSE)
        size_t scratch0 = 0;
        cufftHandle plan0 = 0;
        CUFFT_CHECK(cufftCreate(&plan0));
        CUFFT_CHECK(cufftMpAttachComm(plan0, cufftMpCommType::CUFFT_COMM_NONE, nullptr));
        CUFFT_CHECK(cufftXtSetSubformatDefault(plan0, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
        CUFFT_CHECK(cufftSetAutoAllocation(plan0, 0));
        if (nz == 1) {
          CUFFT_CHECK(cufftMakePlan2d(plan0, nx, ny, CUFFT_C2C, &scratch0));
        } else {
          CUFFT_CHECK(cufftMakePlan3d(plan0, nx, ny, nz, CUFFT_C2C, &scratch0));
        }

        // Create plan #2
        // This plan will be used for (Slabs_Y + CUFFT_FORWARD or Slabs_X + CUFFT_INVERSE)
        size_t scratch1 = 0;
        cufftHandle plan1 = 0;
        CUFFT_CHECK(cufftCreate(&plan1));
        CUFFT_CHECK(cufftMpAttachComm(plan1, cufftMpCommType::CUFFT_COMM_NONE, nullptr));
        CUFFT_CHECK(cufftXtSetSubformatDefault(plan1, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE_SHUFFLED, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE));
        CUFFT_CHECK(cufftSetAutoAllocation(plan1, 0));
        if (nz == 1) {
          CUFFT_CHECK(cufftMakePlan2d(plan1, nx, ny, CUFFT_C2C, &scratch1));
        } else {
          CUFFT_CHECK(cufftMakePlan3d(plan1, nx, ny, nz, CUFFT_C2C, &scratch1));
        }

        std::int64_t count = nx * ny * nz / num_pes;
        size_t scratch = std::max<size_t>(scratch0, scratch1);

        float2* inout_d = (float2*)nvshmem_malloc(count * sizeof(float2));
        float2* scratch_d = (float2*)nvshmem_malloc(count * sizeof(float2));
        CUDA_CHECK(cudaGetLastError());

        CUFFT_CHECK(cufftSetWorkArea(plan0, scratch_d));
        CUFFT_CHECK(cufftSetWorkArea(plan1, scratch_d));

        return std::make_unique<plan_cache>(nx, ny, nz, count, plan0, plan1, inout_d, scratch_d);
    }

    ~plan_cache() {
        // The context is already destroyed at this point, so releasing resources is pointless
        // CUFFT_CHECK(cufftDestroy(plan_inplace_to_shuffled));
        // CUFFT_CHECK(cufftDestroy(plan_shuffled_to_inplace));
        // nvshmem_free(inout_d);
        // nvshmem_free(scratch_d);
        // nvshmem_finalize();
    }

};

// This cache holds a plan for a specific (nx, ny, nz) shape
static std::unique_ptr<plan_cache> cache(nullptr);

// Prevents accidental access by multiple threads to the cache.
// Note that NVSHMEM does not support >1 GPU per process, and cuFFTMp has the same restriction.
static std::mutex plan_mtx;

inline void apply_cufftmp(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {

    /**
     * Extract the parameters of the FFT
     */
    const cufftmpDescriptor &d = *UnpackDescriptor<cufftmpDescriptor>(opaque, opaque_len);
    const std::int64_t nx = d.global_x;
    const std::int64_t ny = d.global_y;
    const std::int64_t nz = d.global_z;
    const int distribution = d.distribution;
    const int direction = d.direction;

    const void *input_d = reinterpret_cast<const void *>(buffers[0]);
    void *output_d = reinterpret_cast<void *>(buffers[1]);

    /**
     * Create a cuFFTMp plan, or fetch one from the cache
     */
    std::lock_guard<std::mutex> lock(plan_mtx);

    if(cache == nullptr) {
        cache = plan_cache::create(nx, ny, nz);
    } else {
        if(cache->nx != nx || cache->ny != ny || cache->nz != nz) {
            throw std::runtime_error("Invalid sizes");
        }
    }

    cufftHandle plan = 0;
    // CUFFT_FORWARD + CUFFT_XT_FORMAT_INPLACE
    // or CUFFT_INVERSE + CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    if( (direction == 0 && distribution == 0) ||  
        (direction == 1 && distribution == 1) ) { 
            plan = cache->plan_inplace_to_shuffled; 
    // Otherwise...
    } else {
            plan = cache->plan_shuffled_to_inplace;
    }

    /**
     * Set streams
     */
    CUFFT_CHECK(cufftSetStream(plan, stream));

    /**
     * Local copy: input_d --> inout_d
     * Execute the FFT in place, from and to NVSHMEM allocate memory: inout_d --> inout_d
     * Local copy: inout_d --> output_d
     */

    // Copy input buffer in inout_d, which is NVSHMEM-allocated
    size_t buffer_size_B = cache->count * sizeof(float2);
    CUDA_CHECK(cudaMemcpyAsync(cache->inout_d, input_d, buffer_size_B, cudaMemcpyDefault, stream));

    // Run the cuFFTMp plan in inout_d
    CUFFT_CHECK(cufftExecC2C(plan, cache->inout_d, cache->inout_d, direction == 0 ? CUFFT_FORWARD : CUFFT_INVERSE));

    // Copy from inout_d to output buffer
    CUDA_CHECK(cudaMemcpyAsync(output_d, cache->inout_d, buffer_size_B, cudaMemcpyDefault, stream));
}

}  // namespace

void gpu_cufftmp(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    apply_cufftmp(stream, buffers, opaque, opaque_len);
}

}  // namespace cufftmp_jax
