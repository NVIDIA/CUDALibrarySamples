#!/usr/bin/env bash
# Run all cuEST C API examples.
#
# Usage:
#   ./run_all.sh [BUILD_DIR]
#
# Defaults:
#   BUILD_DIR  ./build
#
# Environment variables:
#   CUEST_LIB_DIR  Path to cuEST libraries (auto-detected from BUILD_DIR/CMakeCache.txt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-$SCRIPT_DIR/build}"
DATA_DIR="$(cd "$SCRIPT_DIR/../data" 2>/dev/null && pwd || true)"

if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found: $BUILD_DIR"
    echo "Build first: mkdir build && cd build && cmake .. -DCUEST_INCLUDE_DIR=<include> -DCUEST_LIB_DIR=<lib> && make -j"
    exit 1
fi

if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $SCRIPT_DIR/../data"
    exit 1
fi

# Auto-detect CUEST_LIB_DIR from CMakeCache if not provided
if [ -z "${CUEST_LIB_DIR:-}" ]; then
    CACHE="$BUILD_DIR/CMakeCache.txt"
    if [ -f "$CACHE" ]; then
        CUEST_LIB_DIR=$(grep -m1 "^CUEST_LIB_DIR" "$CACHE" | cut -d= -f2)
    fi
fi

if [ -n "${CUEST_LIB_DIR:-}" ]; then
    CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1 || true)
    if [ -n "$CUDA_MAJOR" ] && [ -d "${CUEST_LIB_DIR}/${CUDA_MAJOR}" ]; then
        export LD_LIBRARY_PATH="${CUEST_LIB_DIR}/${CUDA_MAJOR}:${CUEST_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="${CUEST_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
fi

# Input data
XYZ="$DATA_DIR/geometry/h2o.xyz"
XYZ_ECP="$DATA_DIR/geometry/ch2i2.xyz"
GBS="$DATA_DIR/basis_set/def2-svp.gbs"
AUX_GBS="$DATA_DIR/basis_set/def2-universal-jkfit.gbs"
ECP_GBS="$DATA_DIR/basis_set/def2-svp-ecp.gbs"

PASS=0
FAIL=0
SKIP=0

run() {
    local name="$1"
    shift
    local exe="$BUILD_DIR/$name"

    if [ ! -f "$exe" ]; then
        echo "[ SKIP ] $name  (not built)"
        (( SKIP++ )) || true
        return
    fi

    printf "[ RUN  ] %s\n" "$name"
    if "$exe" "$@"; then
        echo "[ PASS ] $name"
        (( PASS++ )) || true
    else
        echo "[ FAIL ] $name"
        (( FAIL++ )) || true
    fi
    echo
}

echo "============================================================"
echo " cuEST C API Examples"
echo " Build dir : $BUILD_DIR"
echo " Data dir  : $DATA_DIR"
echo "============================================================"
echo

# 0_context
run examples/0_context/basic_usage/basic_usage
run examples/0_context/basic_multistream_usage/basic_multistream_usage
run examples/0_context/basic_multigpu_usage/basic_multigpu_usage
run examples/0_context/user_owned_resources/user_owned_resources

# 1_basic_data_structures
run examples/1_basic_data_structures/ao_basis/ao_basis
run examples/1_basic_data_structures/ao_shells/ao_shell
run examples/1_basic_data_structures/ao_basis_general/ao_basis_general    "$XYZ" "$GBS"
run examples/1_basic_data_structures/ecp_shells/ecp_shell
run examples/1_basic_data_structures/ecp_atom/ecp_atom                    "$XYZ_ECP" "$ECP_GBS"
run examples/1_basic_data_structures/xc_grid/xc_grid
run examples/1_basic_data_structures/xc_grid_general/xc_grid_general      "$XYZ"

# 2_one_electron_integrals
run examples/2_one_electron_integrals/one_electron_integrals/one_electron_integrals  "$XYZ" "$GBS"
run examples/2_one_electron_integrals/one_electron_gradients/one_electron_gradients  "$XYZ" "$GBS"

# 3_density_fitting
run examples/3_density_fitting/core_df_jk/core_df_jk                          "$XYZ" "$GBS" "$AUX_GBS"
run examples/3_density_fitting/core_df_jk_gradients/core_df_jk_gradients      "$XYZ" "$GBS" "$AUX_GBS"

# 4_exchange_correlation
run examples/4_exchange_correlation/local_xc_potential/local_xc_potential      "$XYZ" "$GBS"
run examples/4_exchange_correlation/local_xc_gradient/local_xc_gradient        "$XYZ" "$GBS"
run examples/4_exchange_correlation/nonlocal_xc_potential/nonlocal_xc_potential "$XYZ" "$GBS"
run examples/4_exchange_correlation/nonlocal_xc_gradient/nonlocal_xc_gradient  "$XYZ" "$GBS"

# 5_effective_core_potentials
run examples/5_effective_core_potentials/ecp_integrals/ecp_integrals  "$XYZ_ECP" "$GBS" "$ECP_GBS"
run examples/5_effective_core_potentials/ecp_gradients/ecp_gradients  "$XYZ_ECP" "$GBS" "$ECP_GBS"

echo "============================================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================================"

[ "$FAIL" -eq 0 ]
