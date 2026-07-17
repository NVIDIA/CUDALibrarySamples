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

#pragma once

#include <algorithm>
#include <chrono>
#include <thread>

#include <nvbench/criterion_manager.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

// Stopping criterion that runs until the accumulated GPU time exceeds a configurable threshold (default 2 s).
// Note: "total-time" bounds the accumulated GPU (measurement) time, not wall-clock. Because the criterion sleeps
// between iterations (see do_add_measurement), a run's wall-clock duration exceeds the --total-time value, but the
// per-iteration sleep is capped at "max-sleep-time" (default 0.5 s) so a single slow iteration cannot balloon the
// wall-clock time.
// CLI overrides (the criterion must be named first so NVBench accepts its args):
//   --stopping-criterion total-time-criterion --total-time 10 --max-sleep-time 0.5
class total_time_criterion final : public nvbench::stopping_criterion_base
{
  nvbench::float64_t total_time_s_{};

public:
  total_time_criterion()
      : nvbench::stopping_criterion_base{
          "total-time-criterion",
          {{"total-time", nvbench::float64_t{2.0}}, {"max-sleep-time", nvbench::float64_t{0.5}}}
        }
  {}

protected:
  void do_initialize() override { total_time_s_ = 0.0; }

  void do_add_measurement(nvbench::float64_t measurement) override
  {
    total_time_s_ += measurement;

    // Sleep for as long as the iteration took, but no longer than the "max-sleep-time" parameter.
    if (measurement > 0.0)
    {
      const auto sleep_time_s = std::min(measurement, m_params.get_float64("max-sleep-time"));
      std::this_thread::sleep_for(std::chrono::duration<nvbench::float64_t>(sleep_time_s));
    }
  }

  bool do_is_finished() override { return total_time_s_ >= m_params.get_float64("total-time"); }
};
