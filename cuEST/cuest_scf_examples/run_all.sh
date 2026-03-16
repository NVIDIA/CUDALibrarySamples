#!/usr/bin/env bash
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
run_pytest "rhf_1"            test/rhf_1/test.py
run_pytest "rhf_grad_1"       test/rhf_grad_1/test.py
run_pytest "rhf_pcm"          test/rhf_pcm/test.py
run_pytest "b3lyp1_grad_1"    test/b3lyp1_grad_1/test.py
run_pytest "b97mv_grad_1"     test/b97mv_grad_1/test.py
run_pytest "blyp_grad_grids_1" test/blyp_grad_grids_1/test.py

# Standalone examples
run_script "rhf-1 example"   examples/rhf-1/test.py

echo "============================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================================"

[ "$FAIL" -eq 0 ]
