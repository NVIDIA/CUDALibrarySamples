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

#include <cstdio>
#include <cstdlib>
#include <nccl.h>

#define NCCL_CHECK(ans) { nccl_check((ans), __FILE__, __LINE__); }
inline void nccl_check(ncclResult_t code, const char *file, int line) {
    if (code != ncclSuccess) {
        fprintf(stderr, "NCCL_CHECK: %s %s %d\n", ncclGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}
