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

# Run all cuEST SCF examples (tests and standalone examples).
#
# Usage:
#   ./run_all.sh [PYTHON]
#
# Defaults:
#   PYTHON  python3 (or the active virtualenv python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${1:-python3}"

PASS=0
FAIL=0

run_pytest() {
    local label="$1"
    local test_path="$SCRIPT_DIR/$2"

    printf "[ RUN  ] %s\n" "$label"
    if (cd "$SCRIPT_DIR" && "$PYTHON" -m pytest "$test_path" -v --tb=short 2>&1); then
        echo "[ PASS ] $label"
        (( PASS++ )) || true
    else
        echo "[ FAIL ] $label"
        (( FAIL++ )) || true
    fi
    echo
}

run_script() {
    local label="$1"
    local script_path="$SCRIPT_DIR/$2"

    printf "[ RUN  ] %s\n" "$label"
    if (cd "$(dirname "$script_path")" && PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$(basename "$script_path")" 2>&1); then
        echo "[ PASS ] $label"
        (( PASS++ )) || true
    else
        echo "[ FAIL ] $label"
        (( FAIL++ )) || true
    fi
    echo
}

echo "============================================================"
echo " cuEST SCF Examples"
echo " Python : $($PYTHON --version)"
echo " Dir    : $SCRIPT_DIR"
echo "============================================================"
echo

# SCF tests
run_pytest "rhf_1"                test/rhf_1
run_pytest "rhf_grad_1"           test/rhf_grad_1
run_pytest "rhf_pcm"              test/rhf_pcm
run_pytest "rhf_polarizability_1" test/rhf_polarizability_1
run_pytest "uhf_1"                test/uhf_1
run_pytest "b3lyp1_grad_1"        test/b3lyp1_grad_1
run_pytest "b97mv_grad_1"         test/b97mv_grad_1
run_pytest "blyp_grad_grids_1"    test/blyp_grad_grids_1
run_pytest "dft_energies"         test/dft_energies
run_pytest "ecp_1"                test/ecp_1
run_pytest "mp2_1"                test/mp2_1

# Standalone examples
run_script "rhf-1 example"   examples/rhf-1/test.py
run_script "uhf-1 example"   examples/uhf-1/test.py
run_script "mp2-1 example"   examples/mp2-1/test.py
run_script "cphf-1 example"  examples/cphf-1/test.py

echo "============================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================================"

[ "$FAIL" -eq 0 ]
