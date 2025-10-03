/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "omp.h"

#ifdef _WIN32
#error                                                                                   \
    "This sample does not support Windows right now (replace dlopen()/dlsym() with equivalent Windows API functions)"
#endif

#include <dlfcn.h> // required for dlopen and dlsym; only works on Linux

#include "cudss_threading_interface.h"


int main(int argc, char *argv[]) {
    int passed = 1;

    /* Parsing the threading layer information from the input parameters */
    char thr_backend_name[1024];
    char thr_layer_libname[1024];
    if (argc > 2) {
        strcpy(thr_backend_name, argv[1]);
        printf("Threading backend name is: %s\n", thr_backend_name);
        strcpy(thr_layer_libname, argv[2]);
        printf("Threading layer library name is: %s\n", thr_layer_libname);
        fflush(0);
    } else {
        printf("Error: this example requires passing at least two arguments:\n"
               "the threading backend name (openmp)\n"
               "and the threading layer library (full name with the path)\n");
        fflush(0);
        return -2;
    }

    /*
     * TEST No.1: Calling a threading runtime API (without using the cuDSS threading
     * layer)
     */
    int old_num_threads = omp_get_max_threads();
    int new_num_threads = old_num_threads > 1 ? old_num_threads / 2 : 2;
    printf("Changing the number of threads from %d to %d\n", old_num_threads,
           new_num_threads);
    omp_set_num_threads(new_num_threads);
    printf("Number of threads is: %d\n", omp_get_max_threads());
    fflush(0);
    if (omp_get_max_threads() != new_num_threads) {
        printf("Test No.1 [Calling a threading runtime API (without using the cuDSS "
               "threading layer)] FAILED: the number of threads has not changed\n");
        passed = -3;
    } else {
        printf("Test No.1 [Calling a threading runtime API (without using the cuDSS "
               "threading layer)] PASSED\n");
    }
    omp_set_num_threads(old_num_threads);
    fflush(0);

    /*
     * TEST No.2: Calling a threading API from the cuDSS threading layer library
     */
    cudssThreadingInterface_t *thrIface    = NULL;
    void                      *thrIfaceLib = NULL;

    if (passed == 1) {
        thrIfaceLib = static_cast<void *>(dlopen(thr_layer_libname, RTLD_NOW));
        if (thrIfaceLib == NULL) {
            printf("Error: failed to open the threading layer library %s\n",
                   thr_layer_libname);
            fflush(0);
            passed = -4;
        }

        thrIface =
            (cudssThreadingInterface_t *)dlsym(thrIfaceLib, "cudssThreadingInterface");

        if (thrIface == NULL) {
            printf("Error: failed to find the symbol cudssThreadingInterface_t in the "
                   "threading layer library %s\n",
                   thr_layer_libname);
            fflush(0);
            passed = -5;
        }
    }

    int num_threads = omp_get_max_threads();

    if (passed == 1) {
        printf("Querying the number of threads from the threading layer library\n");
        int num_threads_out = thrIface->cudssGetMaxThreads();
        printf("Number of threads is: %d\n", num_threads_out);
        fflush(0);
        if (num_threads_out != num_threads) {
            printf("Test No.2 [Calling a threading API from the cuDSS threading layer "
                   "library] FAILED: the number of threads does not match the expected "
                   "value\n");
            fflush(0);
            passed = -6;
        } else {
            printf("Test No.2 [Calling a threading API from the cuDSS threading layer "
                   "library] PASSED\n");
            fflush(0);
        }
    }

    /*
     * TEST No.3: Calling a parallelFor API from the cuDSS threading layer library
     */
    int nthreads_test3 = 5;
    thrIface->cudssParallelFor(
        nthreads_test3, 10 /*ntasks*/, NULL /*ctx*/, [](int i, void *ctx) {
            printf("Thread %d is working on index %d\n", omp_get_thread_num(), i);
            fflush(0);
        });

    // This test is a bit loose as we can only visually inspect the output and confirm
    // that multiple workers have participated in a parallelFor loop (how many might
    // depend on the threading runtime settings)
    printf("Test No.3 [Calling a parallelFor API from the cuDSS threading layer library] "
           "PASSED\n");
    fflush(0);

    // Cleanup
    if (thrIfaceLib != NULL) {
        dlclose(thrIfaceLib);
    }

    if (passed == 1) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }
    fflush(0);
    // Only return 0 if all tests passed
    return passed == 1 ? 0 : passed;
}
