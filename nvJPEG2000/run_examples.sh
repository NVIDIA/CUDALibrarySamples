#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: this only checks that the build and the setup are valid, i.e. that the examples run against
# the package and produce consistent output. It is not a test of nvJPEG2000 itself.
#
# Runs the installed examples on their images against an nvJPEG2000 package and checks
# what they write, stopping at the first failure. libcudart.so.<CUDA major> must be loadable.
#
#   run_examples.sh <nvjpeg2k package root> <CUDA major>

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <nvjpeg2k-package-root> <cuda-major>" >&2
  exit 2
fi

examples_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_dir="$(cd "$1" && pwd)/lib/$2"
if [[ ! -e "${lib_dir}/libnvjpeg2k.so.0" ]]; then
  echo "Missing nvJPEG2000 library: ${lib_dir}/libnvjpeg2k.so.0" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

bin="${examples_root}/bin"
images="${examples_root}/images"
output="${examples_root}/output"

run() {
  echo "-- $*"
  "$@"
}

check_same() {
  echo "-- check $1 matches $2"
  if [[ ! -s "$1" || ! -s "$2" ]]; then
    echo "Missing or empty: $1 or $2" >&2
    return 1
  fi
  if [[ "$(sha256sum < "$1")" != "$(sha256sum < "$2")" ]]; then
    echo "$1 differs from $2" >&2
    return 1
  fi
}

# Every example and at least one decoder image must be installed, or nothing would be checked.
for example in nvjpeg2k_decode nvjpeg2k_dec_pipelined nvjpeg2k_decode_tile_partial nvjpeg2k_encode; do
  if [[ ! -x "${bin}/${example}" ]]; then
    echo "Missing example: ${bin}/${example}" >&2
    exit 1
  fi
done
if [[ -z "$(find "${images}/decoder" -name '*.jp2' -print -quit 2>/dev/null)" ]]; then
  echo "No .jp2 images in ${images}/decoder" >&2
  exit 1
fi

# Start from empty output directories; the samples do not create them.
rm -rf "${output:?}"
mkdir -p "${output}"/{decode,pipelined,tile_partial,encode,encode_ht,roundtrip,roundtrip_ht}

while IFS= read -r -d '' input; do
  name="$(basename "${input}" .jp2)"
  run "${bin}/nvjpeg2k_decode" -i "${input}" -o "${output}/decode"
  run "${bin}/nvjpeg2k_dec_pipelined" -i "${input}" -o "${output}/pipelined"
  run "${bin}/nvjpeg2k_decode_tile_partial" -i "${input}" -da 0,0,512,512 -o "${output}/tile_partial"

  # Both decoders make the same library calls, so their outputs must be identical.
  check_same "${output}/decode/${name}.bmp" "${output}/pipelined/${name}.bmp"
  # Nothing else checks the partial decode, so require its output.
  echo "-- check ${output}/tile_partial/${name}.bmp was written"
  if [[ ! -s "${output}/tile_partial/${name}.bmp" ]]; then
    echo "Missing or empty: ${output}/tile_partial/${name}.bmp" >&2
    exit 1
  fi
done < <(find "${images}/decoder" -name '*.jp2' -print0 | sort -z)

# The encoder defaults to the reversible wavelet, so decoding its output must reproduce the
# source image exactly, for both block coders.
for suffix in "" _ht; do
  coder_option=()
  if [[ "${suffix}" == _ht ]]; then
    coder_option=(-ht)
  fi
  run "${bin}/nvjpeg2k_encode" -i "${images}/encoder" "${coder_option[@]}" -o "${output}/encode${suffix}"
  run "${bin}/nvjpeg2k_decode" -i "${output}/encode${suffix}" -o "${output}/roundtrip${suffix}"

  # The encoder names its output after its options, so take whatever the decode wrote.
  decoded=("${output}/roundtrip${suffix}"/*)
  if [[ ${#decoded[@]} -ne 1 ]]; then
    echo "Expected one decoded image in ${output}/roundtrip${suffix}, found ${#decoded[@]}" >&2
    exit 1
  fi
  check_same "${decoded[0]}" "${images}/encoder/TestImage640x480.bmp"
done

echo "-- all examples passed"
