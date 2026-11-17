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

#ifndef CUBLASDX_EXAMPLE_PIPELINE_BUDGET_HPP_
#define CUBLASDX_EXAMPLE_PIPELINE_BUDGET_HPP_

#include <cublasdx.hpp>

namespace example {

    // Shared memory available for pipeline depth budgeting. On SM107 device_info reports the oversized shared
    // memory limit, but pipelines are budgeted against the regular 227 KiB, as the library does.
    template<unsigned Arch>
    constexpr unsigned pipeline_shared_memory_budget() {
        return Arch == 1070 ? 227u * 1024u : commondx::device_info<Arch>::shared_memory();
    }

} // namespace example

#endif // CUBLASDX_EXAMPLE_PIPELINE_BUDGET_HPP_
