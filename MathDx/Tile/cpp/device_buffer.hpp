// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "error_checks.hpp"

namespace mathdx_examples {

// Owns a device allocation used by an example. It keeps CUDA setup code small
// without obscuring the explicit copies and kernel launches in each example.
template <class Element>
class device_buffer {
public:
    explicit device_buffer(size_t element_count) : element_count_(element_count) {
        CUDA_CHECK(cudaMalloc(&data_, element_count_ * sizeof(Element)));
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    ~device_buffer() {
        if (data_ != nullptr) CUDA_CHECK(cudaFree(data_));
    }

    void copy_from_host(const Element* source, size_t element_count) {
        check_count(element_count);
        CUDA_CHECK(cudaMemcpy(data_, source, element_count * sizeof(Element), cudaMemcpyHostToDevice));
    }

    void copy_to_host(Element* destination, size_t element_count) const {
        check_count(element_count);
        CUDA_CHECK(cudaMemcpy(destination, data_, element_count * sizeof(Element), cudaMemcpyDeviceToHost));
    }

    Element* data() const { return data_; }

private:
    void check_count(size_t element_count) const {
        if (element_count != element_count_) {
            std::fprintf(
                stderr, "device buffer copy has %zu elements; allocation has %zu\n", element_count, element_count_);
            std::abort();
        }
    }

    Element* data_ = nullptr;
    size_t element_count_ = 0;
};

} // namespace mathdx_examples
