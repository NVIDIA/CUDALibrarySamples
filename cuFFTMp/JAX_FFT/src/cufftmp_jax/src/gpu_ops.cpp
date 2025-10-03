/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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