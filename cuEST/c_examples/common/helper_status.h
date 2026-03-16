/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMON_HELPER_STATUS
#define COMMON_HELPER_STATUS

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This helper provides a "cuestCheck" macro that will print a 
 * resonable error message and terminate the program if a call
 * that returns a cuestStatus_t returns anything other than 
 * CUEST_STATUS_SUCCESS.
 */

#ifdef CUESTAPI
static const char *cuestGetErrorEnum(cuestStatus_t error) {
    switch (error) {
        case CUEST_STATUS_SUCCESS:
            return "CUEST_STATUS_SUCCESS";
        case CUEST_STATUS_EXCEPTION:
            return "CUEST_STATUS_EXCEPTION";
        case CUEST_STATUS_NULL_POINTER:
            return "CUEST_STATUS_NULL_POINTER";
        case CUEST_STATUS_INVALID_ARGUMENT:
            return "CUEST_STATUS_INVALID_ARGUMENT";
        case CUEST_STATUS_INVALID_SIZE:
            return "CUEST_STATUS_INVALID_SIZE";
        case CUEST_STATUS_INVALID_TYPE:
            return "CUEST_STATUS_INVALID_TYPE";
        case CUEST_STATUS_INVALID_PARAMETER:
            return "CUEST_STATUS_INVALID_PARAMETER";
        case CUEST_STATUS_INVALID_ATTRIBUTE:
            return "CUEST_STATUS_INVALID_ATTRIBUTE";
        case CUEST_STATUS_INVALID_HANDLE:
            return "CUEST_STATUS_INVALID_HANDLE";
        case CUEST_STATUS_UNKNOWN_ERROR:
            return "CUEST_STATUS_UNKNOWN_ERROR";
        case CUEST_STATUS_UNSUPPORTED_ARGUMENT:
            return "CUEST_STATUS_UNSUPPORTED_ARGUMENT";
    }

    return "<unknown>";
}

static void cuestCheck(cuestStatus_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "cuEST error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            (unsigned int) result, cuestGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCuestErrors(val) cuestCheck((val), #val, __FILE__, __LINE__)
#endif

#ifdef __cplusplus
} 
#endif

#endif /* COMMON_HELPER_STATUS */
