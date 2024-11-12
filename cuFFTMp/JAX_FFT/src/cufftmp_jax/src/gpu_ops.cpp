#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace cufftmp_jax;

/**
 * Boilerplate used to
 * (1) Expose the gpu_cufftmp function to Python (to launch our custom op)
 * (2) Expose the cufftmpDescriptor (to pass parameters from Python to C++)
 */

namespace {

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["gpu_cufftmp"] = EncapsulateFunction(gpu_cufftmp);
    return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
    m.def("registrations", &Registrations);
    m.def("build_cufftmp_descriptor",
        [](std::int64_t x, std::int64_t y, std::int64_t z, int dist, int dir) { 
            return PackDescriptor(cufftmpDescriptor{x, y, z, dist, dir}); 
        }
    );
}

}  // namespace
