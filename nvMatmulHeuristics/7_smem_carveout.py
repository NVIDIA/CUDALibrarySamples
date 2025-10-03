#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import ctypes
import sys

from nvMatmulHeuristics import (
    NvMatmulHeuristicsInterface,
    NvMatmulHeuristicsTarget,
    NvMatmulHeuristicsFlags,
    NvMatmulHeuristicsMatmulLayout,
    NvMatmulHeuristicsNvidiaGpu,
    NvMatmulHeuristicsBackendProperty,
    layoutToStr
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare GEMM configurations with and without SMEM carveout'
    )

    parser.add_argument('-M', '--m-dim',
                        type=int,
                        required=True,
                        help='Output matrix height')

    parser.add_argument('-N', '--n-dim',
                        type=int,
                        required=True,
                        help='Output matrix width')

    parser.add_argument('-K', '--k-dim',
                        type=int,
                        required=True,
                        help='Reduced dimension')

    parser.add_argument('--gpu',
                        type=str,
                        choices=[gpu.name for gpu in NvMatmulHeuristicsNvidiaGpu if gpu.name != 'END'],
                        required=True,
                        help='Target GPU')

    parser.add_argument('--layout',
                        type=str,
                        choices=[layout.name for layout in NvMatmulHeuristicsMatmulLayout if layout.name != 'END'],
                        default='NN_ROW_MAJOR',
                        help='Matrix layout (default: NN_ROW_MAJOR)')

    parser.add_argument('--backend',
                        type=str,
                        choices=[backend.name for backend in NvMatmulHeuristicsTarget if backend.name != 'END'],
                        default='CUTLASS3',
                        help='Target backend (default: CUTLASS3)')

    parser.add_argument('--precision',
                        type=str,
                        default='HSH',
                        help='Precision string (e.g. HSS, TST) (default: HSH)')

    parser.add_argument('--smem-carveout',
                        type=int,
                        required=True,
                        help='SMEM carveout size in bytes')

    parser.add_argument('--lib-path',
                        type=str,
                        default=None,
                        help='Path to nvMatmulHeuristics shared library (default: uses the library from the wheel)')

    return parser.parse_args()


def print_kernel_config(config, title):
    print(f"\n{title}:")
    print(f"  CTA Tile: {config['kernel'].cta_tile_m}x{config['kernel'].cta_tile_n}x{config['kernel'].cta_tile_k}")
    print(f"  Warp Tile: {config['kernel'].warp_tile_m}x{config['kernel'].warp_tile_n}x{config['kernel'].warp_tile_k}")
    print(f"  Instruction Tile: {config['kernel'].instr_tile_m}x{config['kernel'].instr_tile_n}x{config['kernel'].instr_tile_k}")
    print(f"  Split K: {config['kernel'].split_k}")
    print(f"  Stages: {config['kernel'].stages}")
    print(f"  CTA swizzling: {config['kernel'].swizzle_factor}")
    print(f"  CTA Order: {config['kernel'].cta_order}")
    print(f"  Cluster Config: {config['kernel'].cluster_m}x{config['kernel'].cluster_n}")
    print(f"  Estimated Runtime: {config['runtime'] * 1000:.6f} ms")


def main():
    args = parse_args()

    # Convert string arguments to enum values
    gpu = NvMatmulHeuristicsNvidiaGpu[args.gpu]
    layout = NvMatmulHeuristicsMatmulLayout[args.layout]
    backend = NvMatmulHeuristicsTarget[args.backend]

    print(f"\nComparing configurations with and without SMEM carveout:")
    print(f"Problem size: M={args.m_dim}, N={args.n_dim}, K={args.k_dim}")
    print(f"GPU: {gpu.name}")
    print(f"Layout: {layoutToStr(layout)}")
    print(f"Backend: {backend.name}")
    print(f"Precision: {args.precision}")
    print(f"SMEM Carveout: {args.smem_carveout} bytes\n")

    # Initialize nvMatmulHeuristics interface
    try:
        nvMatmulHeuristics = NvMatmulHeuristicsInterface(
            path=args.lib_path,
            backend=backend,
            precision=args.precision,
            flags=NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING
        )
    except OSError as e:
        print(f"Error: Failed to load nvMatmulHeuristics library: {e}")
        print(f"Make sure the library exists at: {args.lib_path}")
        return 1
    except AssertionError:
        print("Error: Version mismatch or unsupported precision")
        return 1

    try:
        # Create and initialize hardware descriptor
        hw_desc = nvMatmulHeuristics.createHardwareDescriptor()
        try:
            # Set the target GPU
            nvMatmulHeuristics.setHardwarePredefinedGpu(hw_desc, gpu)

            # Load internal discovery set for the specified layout
            success = nvMatmulHeuristics.loadInternalDiscoverySet(layout, hw_desc)
            if not success:
                print("Failed to load internal discovery set. Make sure it is available for selected hardware and layout.")
                return 1

            # Create problem object
            problem = nvMatmulHeuristics.makeNvMatmulHeuristicsProblem(
                args.m_dim, args.n_dim, args.k_dim, layout
            )

            # Get configurations without SMEM carveout
            configs_no_carveout = nvMatmulHeuristics.get(problem, 1, hw_desc)
            if not configs_no_carveout:
                print("No configurations found without SMEM carveout")
                return 1

            # Create backend with SMEM carveout
            backend_obj = nvMatmulHeuristics.createBackend(backend)
            try:
                # Set SMEM carveout size as int32_t
                smem_carveout = ctypes.c_int32(args.smem_carveout)
                nvMatmulHeuristics.setBackendValueProperty(
                    backend_obj,
                    NvMatmulHeuristicsBackendProperty.SMEM_CARVEOUT_SIZE,
                    ctypes.byref(smem_carveout),
                    ctypes.sizeof(smem_carveout)
                )

                # Get configurations with SMEM carveout
                configs_with_carveout = nvMatmulHeuristics.getEx(problem, 1, backend_obj, hw_desc)
                if not configs_with_carveout:
                    print("No configurations found with SMEM carveout")
                    return 1

                # Print and compare configurations
                print_kernel_config(configs_no_carveout[0], "Configuration without SMEM carveout")
                print_kernel_config(configs_with_carveout[0], "Configuration with SMEM carveout")

            finally:
                nvMatmulHeuristics.destroyBackend(backend_obj)

        finally:
            nvMatmulHeuristics.destroyHardwareDescriptor(hw_desc)

    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())