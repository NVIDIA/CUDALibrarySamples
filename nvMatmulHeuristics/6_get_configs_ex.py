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
import sys

from nvMatmulHeuristics import (
    NvMatmulHeuristicsInterfaceEx,
    NvMatmulHeuristicsTarget,
    NvMatmulHeuristicsFlags,
    NvMatmulHeuristicsMatmulLayout,
    NvMatmulHeuristicsNvidiaGpu,
    layoutToStr
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get and display GEMM configurations using NvMatmulHeuristicsInterfaceEx'
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

    parser.add_argument('-B', '--batch-size',
                        type=int,
                        default=1,
                        help='Batch size (default: 1)')

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

    parser.add_argument('--count',
                        type=int,
                        default=8,
                        help='Number of configurations to retrieve (default: 8)')

    parser.add_argument('--lib-path',
                        type=str,
                        default=None,
                        help='Path to nvMatmulHeuristics shared library (default: uses the library from the wheel)')

    parser.add_argument('--no-auto-load',
                        action='store_true',
                        help='Disable automatic loading of discovery sets')

    return parser.parse_args()


def print_kernel_config(config, title):
    """Print kernel configuration details."""
    kernel = config["kernel"]
    runtime = config["runtime"]
    print(f"\n{title}:")
    print(f"  Kernel: {kernel}")
    print(f"  Estimated runtime: {runtime * 1000:.6f} ms")


def main():
    args = parse_args()

    # Convert string arguments to enum values
    gpu = NvMatmulHeuristicsNvidiaGpu[args.gpu]
    layout = NvMatmulHeuristicsMatmulLayout[args.layout]
    backend = NvMatmulHeuristicsTarget[args.backend]

    print(f"\nGetting {args.count} configurations for:")
    print(f"Problem size: M={args.m_dim}, N={args.n_dim}, K={args.k_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU: {gpu.name}")
    print(f"Layout: {layoutToStr(layout)}")
    print(f"Backend: {backend.name}")
    print(f"Precision: {args.precision}")
    print(f"Auto-load discovery sets: {not args.no_auto_load}\n")

    # Initialize nvMatmulHeuristics interface
    try:
        # Create interface with the GPU
        nvMatmulHeuristics = NvMatmulHeuristicsInterfaceEx(
            path=args.lib_path,
            backend=backend,
            flags=NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
            load_discovery_implicitly=not args.no_auto_load,
            gpu=gpu
        )

        # Create problem object
        problem = nvMatmulHeuristics.makeNvMatmulHeuristicsProblem(
            args.m_dim, args.n_dim, args.k_dim, layout, args.batch_size
        )

        # Set precision
        precision = args.precision
        print(f"\nGetting configurations with precision ({precision})...")

        # Track loaded discovery sets before the call
        loaded_before = set(nvMatmulHeuristics._loaded_discovery_sets.keys())

        # Get configurations
        configs = nvMatmulHeuristics.get(problem, args.count, precision=precision)

        # Check which discovery sets were loaded
        loaded_after = set(nvMatmulHeuristics._loaded_discovery_sets.keys())
        newly_loaded = loaded_after - loaded_before

        if newly_loaded:
            print("\nImplicitly loaded discovery sets:")
            for key in newly_loaded:
                target, prec, layout = key
                print(f"  - Target: {target.name}, Precision: {prec}, Layout: {layoutToStr(layout)}")
        else:
            print("\nNo new discovery sets were loaded")

        print(f"\nFound {len(configs)} configurations with {precision} precision:")
        for i, config in enumerate(configs, 1):
            print_kernel_config(config, f"Configuration {i}")

    except OSError as e:
        print(f"Error: Failed to load nvMatmulHeuristics library: {e}")
        print(f"Make sure the library exists at: {args.lib_path}")
        return 1
    except AssertionError:
        print("Error: Version mismatch or unsupported precision")
        return 1
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())