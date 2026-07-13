#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Run all cuEST Python API examples.
#
# Usage:
#   ./run_all.sh [PYTHON]
#
# Defaults:
#   PYTHON  python3 (or the active virtualenv python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${1:-python3}"

# Data files (co-located with c_examples/data, one level up)
DATA_DIR="$(cd "$SCRIPT_DIR/../data" 2>/dev/null && pwd || true)"
if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $SCRIPT_DIR/../data"
    echo "Run scripts/create_cuda_samples.py first."
    exit 1
fi

XYZ="$DATA_DIR/geometry/h2o.xyz"
XYZ_ECP="$DATA_DIR/geometry/ch2i2.xyz"
GBS="$DATA_DIR/basis_set/def2-svp.gbs"
AUX_GBS="$DATA_DIR/basis_set/def2-universal-jkfit.gbs"
ECP_GBS="$DATA_DIR/basis_set/def2-svp-ecp.gbs"

PASS=0
FAIL=0

run() {
    local label="$1"
    local example_dir="$SCRIPT_DIR/$2"
    shift 2

    printf "[ RUN  ] %s\n" "$label"
    if (cd "$example_dir" && "$PYTHON" run.py "$@"); then
        echo "[ PASS ] $label"
        (( PASS++ )) || true
    else
        echo "[ FAIL ] $label"
        (( FAIL++ )) || true
    fi
    echo
}

echo "============================================================"
echo " cuEST Python API Examples"
echo " Python    : $($PYTHON --version)"
echo " Data dir  : $DATA_DIR"
echo "============================================================"
echo

# 0_context
run "basic_usage"             0_context/basic_usage
run "basic_multistream_usage" 0_context/basic_multistream_usage
run "user_owned_resources"    0_context/user_owned_resources

# 1_basic_data_structures
run "ao_shells"  1_basic_data_structures/ao_shells
run "ao_basis"   1_basic_data_structures/ao_basis

# 2_one_electron_integrals
run "one_electron_integrals"  2_one_electron_integrals/one_electron_integrals  "$XYZ" "$GBS"
run "one_electron_gradients"  2_one_electron_integrals/one_electron_gradients  "$XYZ" "$GBS"
run "property_integrals"      2_one_electron_integrals/property_integrals      "$XYZ" "$GBS"
run "property_gradients"      2_one_electron_integrals/property_gradients      "$XYZ" "$GBS"

# 3_density_fitting
run "core_df_jk"              3_density_fitting/core_df_jk              "$XYZ" "$GBS" "$AUX_GBS"
run "core_df_jk_gradient_rhf" 3_density_fitting/core_df_jk_gradient_rhf "$XYZ" "$GBS" "$AUX_GBS"
run "core_df_jk_gradient_uhf" 3_density_fitting/core_df_jk_gradient_uhf "$XYZ" "$GBS" "$AUX_GBS"
run "df_mo_integrals"         3_density_fitting/df_mo_integrals         "$XYZ" "$GBS" "$AUX_GBS"
run "nonsymmetric_core_df_k"  3_density_fitting/nonsymmetric_core_df_k  "$XYZ" "$GBS" "$AUX_GBS"

# 4_exchange_correlation
run "local_xc_potential"    4_exchange_correlation/local_xc_potential    "$XYZ" "$GBS"
run "local_xc_gradient"     4_exchange_correlation/local_xc_gradient     "$XYZ" "$GBS"
run "nonlocal_xc_potential" 4_exchange_correlation/nonlocal_xc_potential "$XYZ" "$GBS"
run "nonlocal_xc_gradient"  4_exchange_correlation/nonlocal_xc_gradient  "$XYZ" "$GBS"

# 5_effective_core_potentials
run "ecp_integrals" 5_effective_core_potentials/ecp_integrals "$XYZ_ECP" "$GBS" "$ECP_GBS"
run "ecp_gradients" 5_effective_core_potentials/ecp_gradients "$XYZ_ECP" "$GBS" "$ECP_GBS"

# 6_pcm
run "pcm_potential" 6_pcm/pcm_potential "$XYZ" "$GBS"
run "pcm_gradients" 6_pcm/pcm_gradients "$XYZ" "$GBS"

echo "============================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================================"

[ "$FAIL" -eq 0 ]
