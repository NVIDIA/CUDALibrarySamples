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

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>

#include "nvcomp/cascaded.h"

namespace nvcomp
{

inline bool parse_strict_int(const char *s, int &out)
{
  if (s == nullptr || *s == '\0')
  {
    return false;
  }
  const char *const end = s + std::strlen(s);
  const auto [parsed_end, error] = std::from_chars(s, end, out);
  return error == std::errc{} && parsed_end == end;
}

inline bool apply_cascaded_mode_arg(const char *arg, nvcompBatchedCascadedCompressOpts_t &opts)
{
  if (std::strcmp(arg, "asymmetric") == 0)
  {
    opts.common_opts.mode = NVCOMP_CASCADED_MODE_ASYMMETRIC;
    return true;
  }
  if (std::strcmp(arg, "symmetric") == 0)
  {
    opts.common_opts.mode = NVCOMP_CASCADED_MODE_SYMMETRIC;
    return true;
  }
  std::cerr << "ERROR: Cascaded mode must be asymmetric or symmetric, " << arg << " was passed" << std::endl;
  return false;
}

inline bool apply_cascaded_level_arg(const char *arg, nvcompBatchedCascadedCompressOpts_t &opts)
{
  constexpr int MAX_REPRESENTABLE_LEVEL = std::numeric_limits<uint8_t>::max();
  int level = 0;
  if (!parse_strict_int(arg, level) || level < 0 || level > MAX_REPRESENTABLE_LEVEL)
  {
    std::cerr << "ERROR: Cascaded level must be an integer between 0 and " << MAX_REPRESENTABLE_LEVEL << "."
              << std::endl;
    return false;
  }
  opts.compression_level = static_cast<uint8_t>(level);
  return true;
}

inline bool apply_cascaded_fine_grained_encodings_arg(const char *arg, nvcompBatchedCascadedCompressOpts_t &opts)
{
  std::string_view remaining = arg == nullptr ? "" : arg;
  if (remaining == "none")
  {
    opts.fine_grained_encoding_flags = NVCOMP_CASCADED_FINE_GRAINED_ENCODING_NONE;
    return true;
  }

  uint64_t encodings = NVCOMP_CASCADED_FINE_GRAINED_ENCODING_NONE;
  while (!remaining.empty())
  {
    const size_t separator = remaining.find(',');
    const std::string_view encoding = remaining.substr(0, separator);
    if (encoding == "delta")
    {
      encodings |= NVCOMP_CASCADED_FINE_GRAINED_ENCODING_DELTA;
    }
    else if (encoding == "rle")
    {
      encodings |= NVCOMP_CASCADED_FINE_GRAINED_ENCODING_RLE;
    }
    else if (encoding == "for")
    {
      encodings |= NVCOMP_CASCADED_FINE_GRAINED_ENCODING_FOR;
    }
    else
    {
      std::cerr << "ERROR: Cascaded fine-grained encodings must be none or a comma-separated list containing delta "
                   "rle, and for; the encoding passed was: "
                << encoding << std::endl;
      return false;
    }

    if (separator == std::string_view::npos)
    {
      opts.fine_grained_encoding_flags = encodings;
      return true;
    }
    remaining.remove_prefix(separator + 1);
  }

  std::cerr << "ERROR: Cascaded fine-grained encodings cannot be empty." << std::endl;
  return false;
}

} // namespace nvcomp
