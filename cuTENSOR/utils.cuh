/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUDALIBRARYSAMPLES_CUTENSOR_UTILS_CUH
#define CUDALIBRARYSAMPLES_CUTENSOR_UTILS_CUH

#include <random>
#include <chrono>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <complex>
#include <stdexcept>
#include <functional>
#include <type_traits>

#include <cuda_runtime.h>

// Handle cuTENSOR errors
inline void handle_error( cutensorStatus_t err )
{
    if ( err != CUTENSOR_STATUS_SUCCESS )
        throw std::runtime_error { std::string { cutensorGetErrorString(err) } }; 
} 

// Handle CUDA errors.
inline void handle_error( cudaError_t err )
{
    if ( err != cudaSuccess )
        throw std::runtime_error { std::string { cudaGetErrorString(err) } };
}

// Utils for exception safe CUDA allocations. These allocation functions
// do not call constructors or destructors.
template <typename T>
using cuda_ptr = std::unique_ptr<T,decltype(&cudaFree)>;

template <typename T>
cuda_ptr<T> cuda_alloc( size_t count )
{
    void* p;
    if ( cudaMalloc(&p,sizeof(T)*count) != cudaSuccess ) throw std::bad_alloc {};
    else return cuda_ptr<T> { reinterpret_cast<T*>(p), &cudaFree };
}


template <typename T>
using cuda_host_ptr = std::unique_ptr<T,decltype(&cudaFreeHost)>;

template <typename T>
cuda_host_ptr<T> cuda_host_alloc( size_t count )
{
    void* p;
    if ( cudaMallocHost(&p,sizeof(T)*count) != cudaSuccess ) throw std::bad_alloc {};
    else return cuda_host_ptr<T> { reinterpret_cast<T*>(p), &cudaFreeHost };
}


// Random numbers.
inline
std::mt19937 get_seeded_random_engine() 
{
    using     rand_type = std::random_device::result_type;
    using mersenne_type = std::mt19937::result_type;
    constexpr size_t N  = std::mt19937::state_size * sizeof(mersenne_type);
    constexpr size_t M  =  1 + (N-1)/sizeof(rand_type);

    rand_type random_data[M];
    std::random_device source;
    std::generate(random_data,random_data+M,std::ref(source));
    std::seed_seq seed(random_data,random_data+M);

    return std::mt19937( seed );
}

template <typename T, typename Enable = void>
class randomgen;

// Specialisation for float, double, etc.
template <typename fp>
class randomgen
<
    fp,
    std::enable_if_t< std::is_floating_point_v<fp> > 
>
{
public:
    using distribution = std::uniform_real_distribution<fp>;

    randomgen( fp min = 0, fp max = 1 ):
    r( std::bind(distribution(min,max),get_seeded_random_engine()) )
    {}

    fp operator()() const { return r(); }

private:
    std::function<fp()> r;
};

// Specialisation for integral types.
template <typename integer>
class randomgen
<
    integer,
    std::enable_if_t< std::is_integral_v<integer> > 
>
{
public:
    randomgen( integer min = std::numeric_limits<integer>::min() , 
               integer max = std::numeric_limits<integer>::max() ):
    r( std::bind( std::uniform_int_distribution<>(min,max),
                  get_seeded_random_engine() ) ) 
    {}

    integer operator()() const { return r(); }

private:
    std::function<integer()> r;
};

// Specialisation for complex types. min and max are understood
// separately for real an imaginary parts each.
template <typename value_type>
class randomgen< std::complex<value_type>, void >
{
public:
    using T = std::complex<value_type>;

    randomgen() = default;

    randomgen( T min, T max ):
    re { min.real(), max.real() },
    im { min.imag(), max.imag() }
    {}


    T operator()() const { return T { re(), im() }; }

private:
    randomgen<value_type> re, im;
};





// Timers.
class GPUTimer
{
public:

    GPUTimer( cudaStream_t stream = nullptr ) 
    {
        handle_error(cudaEventCreate(&start_));
        handle_error(cudaEventCreate(&stop_));
        handle_error(cudaEventRecord(start_,stream));
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start( cudaStream_t stream = nullptr ) 
    {
        handle_error(cudaEventRecord(start_,stream));
    }

    float seconds( cudaStream_t stream = nullptr ) 
    {
        handle_error(cudaEventRecord(stop_,stream));
        handle_error(cudaEventSynchronize(stop_));
        float time;
        handle_error(cudaEventElapsedTime(&time, start_, stop_));
        return static_cast<float>(time * 0.001f);
    }

    float stop( cudaStream_t stream = nullptr )
    {
        return seconds(stream);
    }

private:
    cudaEvent_t start_, stop_;

};

class CPUTimer
{
public:
    CPUTimer() { start(); }

    void start()
    {
        start_ = std::chrono::steady_clock::now();
    }

    float seconds()
    {
        end_ = std::chrono::steady_clock::now();
        elapsed_ = end_ - start_;
        return elapsed_.count();
    }

    float stop()
    {
        return seconds();
    }

private:
    std::chrono::steady_clock::time_point start_, end_;
    std::chrono::duration<float> elapsed_;
};




// The remainder of this file is the implementation of finally(), as described in
// the C++ Core Guidelines:
//
//    https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#e19-use-a-final_action-object-to-express-cleanup-if-no-suitable-resource-handle-is-available
//
// The implementation is taken from the ‘Guidelines Support Library’ (GSL):
//
//     https://github.com/microsoft/GSL
//
// finally() allows us to write exception safe C++ code that interfaces
// with plain C APIs. The following copyright notice only applies to the
// implementation of finally().


// Copyright (c) 2015 Microsoft Corporation. All rights reserved. 
// 
// This code is licensed under the MIT License (MIT). 
//
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal  
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
// of the Software, and to permit persons to whom the Software is furnished to do 
// so, subject to the following conditions: 
// 
// The above copyright notice and this permission notice shall be included in all 
// copies or substantial portions of the Software. 
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
// THE SOFTWARE. 

#if defined(__cplusplus) && (__cplusplus >= 201703L)
#define GSL_NODISCARD [[nodiscard]]
#else
#define GSL_NODISCARD
#endif // defined(__cplusplus) && (__cplusplus >= 201703L)

// final_action allows you to ensure something gets run at the end of a scope
template <class F>
class final_action
{
public:
    explicit final_action(const F& ff) noexcept : f{ff} { }
    explicit final_action(F&& ff) noexcept : f{std::move(ff)} { }

    ~final_action() noexcept { if (invoke) f(); }

    final_action(final_action&& other) noexcept
        : f(std::move(other.f)), invoke(std::exchange(other.invoke, false))
    { }

    final_action(const final_action&)   = delete;
    void operator=(const final_action&) = delete;
    void operator=(final_action&&)      = delete;

private:
    F f;
    bool invoke = true;
};

// finally() - convenience function to generate a final_action
template <class F>
GSL_NODISCARD auto finally(F&& f) noexcept
{
    return final_action<std::decay_t<F>>{std::forward<F>(f)};
}

#endif

