/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"

#ifdef _WIN32
#error                                                                                   \
    "This sample does not support Windows right now (replace dlopen()/dlsym() with equivalent Windows API functions)"
#endif

#include <dlfcn.h> // required for dlopen and dlsym; only works on Linux

#include "cudss_threading_interface.h"
#include <library_types.h>

#define CUDSS_THREADING_INTERFACE_MIN_MAJOR_VERSION 0
#define CUDSS_THREADING_INTERFACE_MIN_MINOR_VERSION 8
#define CUDSS_THREADING_INTERFACE_MIN_PATCH_VERSION 0


int main(int argc, char *argv[]) {
    int passed = 1;

    /* Parsing the threading layer information from the input parameters */
    char thr_backend_name[1024];
    char thr_layer_libname[1024];
    if (argc > 2) {
        snprintf(thr_backend_name, sizeof thr_backend_name, "%s", argv[1]);
        printf("Threading backend name is: %s\n", thr_backend_name);
        snprintf(thr_layer_libname, sizeof thr_layer_libname, "%s", argv[2]);
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
    int                        thr_iface_ok = 0;

    if (passed == 1) {
        thrIfaceLib = static_cast<void *>(dlopen(thr_layer_libname, RTLD_NOW));
        if (thrIfaceLib == NULL) {
            printf("Error: failed to open the threading layer library %s\n",
                   thr_layer_libname);
            fflush(0);
            passed = -4;
        } else {
            thrIface =
                (cudssThreadingInterface_t *)dlsym(thrIfaceLib, "cudssThreadingInterface");

            if (thrIface == NULL) {
                printf("Error: failed to find the symbol cudssThreadingInterface_t in the "
                       "threading layer library %s\n",
                       thr_layer_libname);
                fflush(0);
                passed = -5;
            } else if (thrIface->cudssThreadingGetProperty == NULL) {
                printf("Error: cudssThreadingGetProperty is NULL\n");
                fflush(0);
                passed = -7;
            } else {
                int thr_major = 0, thr_minor = 0, thr_patch = 0;
                int prop_err = thrIface->cudssThreadingGetProperty(MAJOR_VERSION, &thr_major);
                if (prop_err != 0) {
                    printf("Error: cudssThreadingGetProperty(MAJOR_VERSION) returned %d\n",
                           prop_err);
                    fflush(0);
                    passed = -7;
                } else if ((prop_err = thrIface->cudssThreadingGetProperty(MINOR_VERSION,
                                                                                    &thr_minor)) != 0) {
                    printf("Error: cudssThreadingGetProperty(MINOR_VERSION) returned %d\n",
                           prop_err);
                    fflush(0);
                    passed = -7;
                } else if ((prop_err = thrIface->cudssThreadingGetProperty(
                                PATCH_LEVEL, &thr_patch)) != 0) {
                    printf("Error: cudssThreadingGetProperty(PATCH_LEVEL) returned %d\n",
                           prop_err);
                    fflush(0);
                    passed = -7;
                } else {
                    const int thr_version_ok =
                        (thr_major > CUDSS_THREADING_INTERFACE_MIN_MAJOR_VERSION) ||
                        (thr_major == CUDSS_THREADING_INTERFACE_MIN_MAJOR_VERSION &&
                         thr_minor > CUDSS_THREADING_INTERFACE_MIN_MINOR_VERSION) ||
                        (thr_major == CUDSS_THREADING_INTERFACE_MIN_MAJOR_VERSION &&
                         thr_minor == CUDSS_THREADING_INTERFACE_MIN_MINOR_VERSION &&
                         thr_patch >= CUDSS_THREADING_INTERFACE_MIN_PATCH_VERSION);
                    if (!thr_version_ok) {
                        printf("Error: threading layer version %d.%d.%d is below minimum required "
                               "%d.%d.%d\n",
                               thr_major, thr_minor, thr_patch,
                               CUDSS_THREADING_INTERFACE_MIN_MAJOR_VERSION,
                               CUDSS_THREADING_INTERFACE_MIN_MINOR_VERSION,
                               CUDSS_THREADING_INTERFACE_MIN_PATCH_VERSION);
                        fflush(0);
                        passed = -7;
                    } else {
                        thr_iface_ok = 1;
                    }
                }
            }
        }
    }

    int num_threads = omp_get_max_threads();

    if (passed == 1 && thr_iface_ok) {
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
    if (thr_iface_ok) {
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
    }

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
