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

#ifndef CUBLASDX_EXAMPLE_CLUSTER_HPP_
#define CUBLASDX_EXAMPLE_CLUSTER_HPP_

#include "common.hpp"

#ifndef CUBLASDX_EXAMPLE_NVRTC

namespace example {

    // Applies the swizzle to clusters and preserves each block's rank within its cluster.
    template<unsigned SwizzleM, unsigned ClusterM, unsigned ClusterN>
    __device__ __forceinline__ cuda::std::tuple<unsigned, unsigned> get_cluster_swizzled_tile_coord() {
        static_assert(SwizzleM > 0, "Threadblock swizzle group must be positive");
        static_assert(ClusterM > 0 and ClusterN > 0, "Cluster extents must be positive");

        unsigned const rank_m = blockIdx.x % ClusterM;
        unsigned const rank_n = blockIdx.y % ClusterN;

        // Grid of clusters, not of CTAs.
        unsigned const clusters_m = gridDim.x / ClusterM;
        unsigned const clusters_n = gridDim.y / ClusterN;
        unsigned const cluster_x  = blockIdx.x / ClusterM;
        unsigned const cluster_y  = blockIdx.y / ClusterN;

        unsigned cluster_m_coord = cluster_x;
        unsigned cluster_n_coord = cluster_y;

        // Keep the swizzle footprint in tiles independent of the cluster shape.
        constexpr unsigned group_tiles_m = (SwizzleM > ClusterM) ? (SwizzleM / ClusterM) : 1u;

        if constexpr (group_tiles_m > 1) {
            unsigned const group_m  = group_tiles_m;
            unsigned const linear   = cluster_x + cluster_y * clusters_m;
            unsigned const group    = linear / (group_m * clusters_n);
            unsigned const first_m  = group * group_m;
            unsigned const active_m = (first_m + group_m <= clusters_m) ? group_m : (clusters_m - first_m);
            unsigned const offset   = linear - group * group_m * clusters_n;
            cluster_m_coord         = first_m + (offset % active_m);
            cluster_n_coord         = offset / active_m;
        }

        return {cluster_m_coord * ClusterM + rank_m, cluster_n_coord * ClusterN + rank_n};
    }

    // Rasterizes over clusters in units of per-CTA output tiles. Each cluster covers
    // a BlocksPerCluster<M, N> block of tiles.
    //
    // Entry i is the tile coordinate of cluster i's top-left CTA. Each CTA adds its
    // own rank inside the cluster, which relies on BlocksPerCluster<>'s layout contract:
    // the cluster is laid out column-major, blockIdx.x M-fastest.
    //
    // The window is measured in tiles and exists for L2 locality; it is divided by the
    // cluster extents so it can be applied in cluster units.
    template<class ClusterCtasMN>
    struct windowed_rasterizer {
        // Scalar state keeps the by-value kernel parameter compact.
        unsigned clusters_m; // cluster-grid extents
        unsigned clusters_n;
        unsigned group_m;    // swizzle group height, in CLUSTERS

        __host__ __device__ unsigned clusters() const { return clusters_m * clusters_n; }

        // Clusters are contiguous along x, consistent with the column-major cluster
        // layout.
        __device__ int begin() const { return blockIdx.x / cute::size<0>(ClusterCtasMN {}); }
        __device__ int stride() const { return gridDim.x / cute::size<0>(ClusterCtasMN {}); }

        // In per-CTA tiles; rank_m/rank_n are this CTA's coordinates inside the
        // cluster (column-major contract: rank_m = blockIdx.x % cluster M extent).
        __device__ auto tile_coord(int const i, unsigned const rank_m, unsigned const rank_n) const {
            // Walk `group_m` cluster rows in column-major order to keep concurrently
            // resident clusters in a compact L2-local region.
            unsigned const linear   = static_cast<unsigned>(i);
            unsigned const per_grp  = group_m * clusters_n;
            unsigned const group    = linear / per_grp;
            unsigned const first_m  = group * group_m;
            unsigned const active_m = (first_m + group_m <= clusters_m) ? group_m : (clusters_m - first_m);
            unsigned const offset   = linear - group * per_grp;

            return cute::make_coord((first_m + (offset % active_m)) * cute::size<0>(ClusterCtasMN {}) + rank_m,
                                    (offset / active_m) * cute::size<1>(ClusterCtasMN {}) + rank_n);
        }
    };

    // tile_grid    : (tiles_m, tiles_n) -- per-CTA tiles, each extent must be an
    //                exact multiple of the corresponding cluster extent
    // cluster_ctas : (blocks_per_cluster_of_v_m, blocks_per_cluster_of_v_n) -- static, BlocksPerCluster<> in CTAs
    // window       : (_16{}, _16{}) -- static, in tiles, sets the swizzle group
    template<class TileGridMN, class ClusterCtasMN, class WindowMN>
    auto build_windowed_rasterizer(TileGridMN const& tile_grid, ClusterCtasMN, WindowMN) {
        static_assert(cute::evenly_divides(WindowMN {}, ClusterCtasMN {}),
                      "raster window must divide by the cluster's CTA extents");

        unsigned const clusters_m =
            static_cast<unsigned>(cute::get<0>(tile_grid) / cute::size<0>(ClusterCtasMN {}));
        unsigned const clusters_n =
            static_cast<unsigned>(cute::get<1>(tile_grid) / cute::size<1>(ClusterCtasMN {}));
        unsigned const group_m =
            static_cast<unsigned>(cute::get<0>(WindowMN {}) / cute::size<0>(ClusterCtasMN {}));

        return windowed_rasterizer<ClusterCtasMN> {
            clusters_m,
            clusters_n,
            (group_m > 0 ? group_m : 1u)};
    }

    // Configures the cluster dimension, dynamic shared memory and the
    // non-portable-size opt-in.
    //
    // Use the cluster dimensions reported by the pipeline whenever uses_cluster is true.
    //
    // Cluster x maps to M and cluster y maps to N. Clusters are contiguous along x,
    // allowing a kernel to recover a linear cluster id from blockIdx.x.
    template<class Pipeline, class Kernel, class... Args>
    cudaError_t launch(Pipeline const&    pipeline,
                       Kernel             kernel,
                       dim3 const         grid,
                       unsigned const     smem_bytes,
                       cudaStream_t const stream,
                       Args&&... args) {
        auto status = cudaFuncSetAttribute(
            reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
        if (status != cudaSuccess) {
            return status;
        }

        cudaLaunchConfig_t config {};
        config.gridDim          = grid;
        config.blockDim         = pipeline.get_block_dim();
        config.dynamicSmemBytes = smem_bytes;
        config.stream           = stream;

        if constexpr (Pipeline::uses_cluster) {
            dim3 const cluster = pipeline.get_cluster_dim();

            if ((cluster.x * cluster.y * cluster.z) > 8u) {
                status = cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel),
                                              cudaFuncAttributeNonPortableClusterSizeAllowed,
                                              1);
                if (status != cudaSuccess) {
                    return status;
                }
            }

            cudaLaunchAttribute attribute[1];
            attribute[0].id               = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = cluster.x;
            attribute[0].val.clusterDim.y = cluster.y;
            attribute[0].val.clusterDim.z = cluster.z;
            config.attrs                  = attribute;
            config.numAttrs               = 1;

            return cudaLaunchKernelEx(&config, kernel, static_cast<Args&&>(args)...);
        } else {
            return cudaLaunchKernelEx(&config, kernel, static_cast<Args&&>(args)...);
        }
    }

    // Query the driver because clusters cannot straddle GPCs and
    // num_sms / cluster_size can overestimate residency.
    template<class Kernel>
    dim3 suggest_persistent_cluster_grid(Kernel kernel, dim3 const block, unsigned const smem, dim3 const cluster) {
        int device = 0;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
        cudaDeviceProp props {};
        CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&props, device));
        int const cluster_size = static_cast<int>(cluster.x * cluster.y * cluster.z);

        // Include the requested dynamic shared-memory limit in the occupancy query.
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        if (cluster_size == 1) {
            int blocks_per_sm = 0;
            CUDA_CHECK_AND_EXIT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, kernel, static_cast<int>(block.x * block.y * block.z), smem));
            int const blocks = blocks_per_sm * props.multiProcessorCount;
            return dim3(static_cast<unsigned>(blocks > 0 ? blocks : 1), 1, 1);
        }

        if (cluster_size > 8) {
            CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
                reinterpret_cast<void const*>(kernel), cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        }

        cudaLaunchConfig_t probe {};
        probe.gridDim          = dim3(cluster.x * (props.multiProcessorCount / cluster_size), cluster.y, cluster.z);
        probe.blockDim         = block;
        probe.dynamicSmemBytes = smem;

        cudaLaunchAttribute attribute[1];
        attribute[0].id               = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster.x;
        attribute[0].val.clusterDim.y = cluster.y;
        attribute[0].val.clusterDim.z = cluster.z;
        probe.attrs                   = attribute;
        probe.numAttrs                = 1;

        int num_clusters = 0;
        CUDA_CHECK_AND_EXIT(cudaOccupancyMaxActiveClusters(&num_clusters, kernel, &probe));
        if (num_clusters < 1) {
            num_clusters = 1;
        }
        return dim3(cluster.x * num_clusters, cluster.y, cluster.z);
    }

} // namespace example

#endif // CUBLASDX_EXAMPLE_NVRTC

#endif // CUBLASDX_EXAMPLE_CLUSTER_HPP_
