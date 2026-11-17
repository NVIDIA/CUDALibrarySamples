# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D low-pass filtered backprojection with cuTile FFT and optional plotting.

Run with ``--plot reconstruction.png`` to save a reconstruction diagnostic
plus a common-scale montage of every axial reconstruction slice.
Matplotlib is required only when that option is used.
"""

import argparse
import math
from pathlib import Path

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft

# A small parallel-beam data set: 256 in-plane views of one asymmetric
# tetrahedron. Each view has a 32x32 detector where u rotates in the x-y plane
# and v is parallel to z. The shape deliberately breaks rotational symmetry,
# making the angle-dependent projections and saved reconstruction diagnostic
# meaningful visual checks.
VOLUME_EXTENT = 16
DETECTOR_EXTENT = 32
PROJECTION_COUNT = 256
TETRAHEDRON_VERTICES = (
    (-4.5, -3.5, -4.0),
    (4.5, -2.5, -3.5),
    (-2.5, 4.5, -2.5),
    (1.5, 1.0, 5.0),
)
ROWS = PROJECTION_COUNT * DETECTOR_EXTENT
Z_SLAB_EXTENT = 4
DETECTOR_V_OFFSET = (DETECTOR_EXTENT - VOLUME_EXTENT) // 2


@ct.kernel
def fused_filtered_backprojection_kernel(
    projections: ct.Array,
    response: ct.Array,
    view_cosines: ct.Array,
    view_sines: ct.Array,
    reconstruction: ct.Array,
):
    """Filter and backproject one z slab without global weight/filter buffers.

    Each CTA filters four detector rows for every view. It constructs the two
    nonzero bilinear weights from the [y, x, u] Tile indices and the view's
    sine/cosine geometry, then reduces u. The filtered rows and weights stay
    in Tile registers throughout the backprojection.
    """
    slab = ct.bid(0)
    load_rows_shape = (1, Z_SLAB_EXTENT, DETECTOR_EXTENT)
    fft_rows_shape = (Z_SLAB_EXTENT, DETECTOR_EXTENT)
    spectrum_shape = (Z_SLAB_EXTENT, DETECTOR_EXTENT, 1)
    response_shape = (1, DETECTOR_EXTENT, 1)
    weights_shape = (1, VOLUME_EXTENT, VOLUME_EXTENT, DETECTOR_EXTENT)
    voxel_slab_shape = (Z_SLAB_EXTENT, VOLUME_EXTENT, VOLUME_EXTENT)
    detector_row_tile = slab + DETECTOR_V_OFFSET // Z_SLAB_EXTENT
    volume_center = ct.float32(0.5 * (VOLUME_EXTENT - 1))
    detector_center = ct.float32(0.5 * (DETECTOR_EXTENT - 1))
    y_coordinates = (
        ct.reshape(ct.arange(VOLUME_EXTENT, dtype=ct.float32), (1, VOLUME_EXTENT, 1, 1))
        - volume_center
    )
    x_coordinates = (
        ct.reshape(ct.arange(VOLUME_EXTENT, dtype=ct.float32), (1, 1, VOLUME_EXTENT, 1))
        - volume_center
    )
    u_indices = ct.reshape(
        ct.arange(DETECTOR_EXTENT, dtype=ct.int32), (1, 1, 1, DETECTOR_EXTENT)
    )
    accumulated = ct.full(voxel_slab_shape, 0.0, ct.float32)

    for projection in range(PROJECTION_COUNT):
        rows = ct.reshape(
            ct.load(
                projections, (projection, detector_row_tile, 0), shape=load_rows_shape
            ),
            fft_rows_shape,
        )
        spectrum = mfft.rfft(rows, axis=1, complex_layout="full")
        response_tile = ct.reshape(
            ct.load(response, (0,), shape=(DETECTOR_EXTENT,)), response_shape
        )
        real = ct.extract(spectrum, (0, 0, 0), shape=spectrum_shape)
        imag = ct.extract(spectrum, (0, 0, 1), shape=spectrum_shape)
        filtered_rows = ct.astype(
            mfft.irfft(
                ct.cat((real * response_tile, imag * response_tile), 2),
                axis=1,
                complex_layout="full",
            ),
            projections.dtype,
        )
        cosine = ct.reshape(
            ct.load(view_cosines, (projection,), shape=(1,)), (1, 1, 1, 1)
        )
        sine = ct.reshape(ct.load(view_sines, (projection,), shape=(1,)), (1, 1, 1, 1))
        detector_u = x_coordinates * cosine + y_coordinates * sine + detector_center
        # All detector coordinates are positive and leave one element of
        # padding, so integer conversion implements floor and u0 + 1 is valid.
        u0 = ct.astype(detector_u, ct.int32)
        du = detector_u - ct.astype(u0, ct.float32)
        du_expanded = ct.zeros(weights_shape, dtype=ct.float32) + du
        weights = ct.where(
            u_indices == u0,
            ct.full(weights_shape, 1.0, dtype=ct.float32) - du_expanded,
            0.0,
        )
        weights = weights + ct.where(u_indices == u0 + 1, du_expanded, 0.0)
        weighted_rows = (
            ct.reshape(
                filtered_rows * ct.float32(1.0 / DETECTOR_EXTENT),
                (Z_SLAB_EXTENT, 1, 1, DETECTOR_EXTENT),
            )
            * weights
        )
        accumulated = accumulated + ct.astype(
            ct.reshape(ct.sum(weighted_rows, 3), voxel_slab_shape), accumulated.dtype
        )

    ct.store(
        reconstruction,
        (slab, 0, 0),
        ct.astype(
            accumulated * ct.float32(math.pi / PROJECTION_COUNT), reconstruction.dtype
        ),
    )


def make_projections() -> cp.ndarray:
    """Return analytic parallel-ray line integrals through the tetrahedron."""
    vertices = cp.asarray(TETRAHEDRON_VERTICES, dtype=cp.float32)
    face_indices = cp.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
    opposite_indices = cp.asarray((0, 1, 2, 3))
    faces = vertices[face_indices]
    face_origins = faces[:, 0]
    normals = cp.cross(faces[:, 1] - face_origins, faces[:, 2] - face_origins)
    inward = cp.sum(normals * (vertices[opposite_indices] - face_origins), axis=1) > 0.0
    normals = cp.where(inward[:, None], -normals, normals)

    detector_coordinates = cp.arange(DETECTOR_EXTENT, dtype=cp.float32)
    detector_coordinates -= 0.5 * (DETECTOR_EXTENT - 1)
    v, u = cp.meshgrid(detector_coordinates, detector_coordinates, indexing="ij")
    angles = math.pi * cp.arange(PROJECTION_COUNT, dtype=cp.float32) / PROJECTION_COUNT
    u_axis = cp.stack((cp.cos(angles), cp.sin(angles), cp.zeros_like(angles)), axis=1)
    ray_direction = cp.stack(
        (-cp.sin(angles), cp.cos(angles), cp.zeros_like(angles)), axis=1
    )
    z_offset = cp.stack((cp.zeros_like(v), cp.zeros_like(v), v), axis=2)
    ray_origin = (
        u[None, :, :, None] * u_axis[:, None, None, :] + z_offset[None, :, :, :]
    )

    direction_coefficient = cp.einsum("pc,fc->pf", ray_direction, normals)
    origin_coefficient = cp.sum(
        (ray_origin[:, :, :, None, :] - face_origins[None, None, None, :, :])
        * normals[None, None, None, :, :],
        axis=4,
    )
    lower = cp.full((PROJECTION_COUNT, DETECTOR_EXTENT, DETECTOR_EXTENT), -cp.inf)
    upper = cp.full_like(lower, cp.inf)
    valid = cp.ones_like(lower, dtype=cp.bool_)
    for face in range(4):
        coefficient = direction_coefficient[:, face].reshape(-1, 1, 1)
        parallel = cp.abs(coefficient) <= 1e-6
        origin_side = origin_coefficient[:, :, :, face]
        valid &= ~parallel | (origin_side <= 0.0)
        boundary = -origin_side / cp.where(
            parallel, cp.ones_like(coefficient), coefficient
        )
        upper = cp.where(coefficient > 1e-6, cp.minimum(upper, boundary), upper)
        lower = cp.where(coefficient < -1e-6, cp.maximum(lower, boundary), lower)
    return cp.where(
        valid & (upper >= lower), cp.maximum(upper - lower, 0.0), cp.zeros_like(lower)
    )


def reference_backprojection(
    projections: cp.ndarray,
    response: cp.ndarray,
    view_cosines: cp.ndarray,
    view_sines: cp.ndarray,
) -> cp.ndarray:
    """Return an independent CuPy filtered-backprojection reference."""
    filtered = cp.fft.ifft(
        cp.fft.fft(projections, axis=2) * response[None, None, :], axis=2
    ).real
    coordinates = cp.arange(VOLUME_EXTENT, dtype=cp.float32) - 0.5 * (VOLUME_EXTENT - 1)
    y_coordinates, x_coordinates = cp.meshgrid(coordinates, coordinates, indexing="ij")
    detector_center = cp.float32(0.5 * (DETECTOR_EXTENT - 1))
    reconstruction = cp.zeros(
        (VOLUME_EXTENT, VOLUME_EXTENT, VOLUME_EXTENT), dtype=cp.float32
    )

    for projection in range(PROJECTION_COUNT):
        detector_u = (
            x_coordinates * view_cosines[projection]
            + y_coordinates * view_sines[projection]
            + detector_center
        )
        u0 = cp.floor(detector_u).astype(cp.int32)
        du = detector_u - u0
        rows = filtered[
            projection, DETECTOR_V_OFFSET : DETECTOR_V_OFFSET + VOLUME_EXTENT
        ]
        reconstruction += (
            rows[:, u0] * (1.0 - du)[None, :, :] + rows[:, u0 + 1] * du[None, :, :]
        )

    return reconstruction * (math.pi / PROJECTION_COUNT)


def save_plot(projections: cp.ndarray, reconstruction: cp.ndarray, path: str) -> None:
    """Save reconstruction summary panels and every axial slice."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "--plot requires matplotlib; install it with `uv pip install matplotlib`"
        ) from error

    output_path = Path(path)
    reconstruction_host = reconstruction.get()
    projection = projections[0].get()
    central_slice = reconstruction_host[VOLUME_EXTENT // 2]
    maximum_intensity = reconstruction_host.max(axis=0)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (projection, central_slice, maximum_intensity),
        ("Projection (angle 0)", "Central z slice", "z maximum-intensity projection"),
    ):
        rendered = axis.imshow(image, cmap="magma", origin="lower")
        axis.set_title(title)
        figure.colorbar(rendered, ax=axis, shrink=0.8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    slice_path = output_path.with_name(
        f"{output_path.stem}_axial_slices{output_path.suffix or '.png'}"
    )
    slice_columns = 4
    slice_rows = math.ceil(VOLUME_EXTENT / slice_columns)
    slice_figure, slice_axes = plt.subplots(
        slice_rows,
        slice_columns,
        figsize=(10, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    slice_minimum = float(reconstruction_host.min())
    slice_maximum = float(reconstruction_host.max())
    for z_index, axis in enumerate(slice_axes.flat):
        if z_index >= VOLUME_EXTENT:
            axis.set_visible(False)
            continue
        rendered = axis.imshow(
            reconstruction_host[z_index],
            cmap="magma",
            origin="lower",
            vmin=slice_minimum,
            vmax=slice_maximum,
        )
        axis.set_title(f"z = {z_index - 0.5 * (VOLUME_EXTENT - 1):.1f}")
    slice_figure.colorbar(
        rendered, ax=slice_axes, shrink=0.8, label="backprojected intensity"
    )
    slice_figure.savefig(slice_path, dpi=160)
    plt.close(slice_figure)
    print(f"saved visual checks to {output_path} and {slice_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", metavar="PATH", help="write diagnostic and all-axial-slices images"
    )
    arguments = parser.parse_args()
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("a CUDA-capable CuPy installation is required")

    projections = make_projections()
    response_bins = cp.arange(DETECTOR_EXTENT, dtype=cp.float32)
    signed_bins = cp.where(
        response_bins <= DETECTOR_EXTENT // 2,
        response_bins,
        response_bins - DETECTOR_EXTENT,
    )
    # Discrete Ram-Lak response in the public full RFFT layout.
    response = cp.abs(signed_bins) / DETECTOR_EXTENT
    angles = math.pi * cp.arange(PROJECTION_COUNT, dtype=cp.float32) / PROJECTION_COUNT
    view_cosines = cp.cos(angles)
    view_sines = cp.sin(angles)
    reconstruction = cp.empty(
        (VOLUME_EXTENT, VOLUME_EXTENT, VOLUME_EXTENT), dtype=cp.float32
    )

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    ct.launch(
        cp.cuda.get_current_stream(),
        (VOLUME_EXTENT // Z_SLAB_EXTENT, 1, 1),
        fused_filtered_backprojection_kernel,
        (projections, response, view_cosines, view_sines, reconstruction),
    )
    stop.record()
    cp.cuda.get_current_stream().synchronize()
    fused_ms = cp.cuda.get_elapsed_time(start, stop)

    reference = reference_backprojection(
        projections, response, view_cosines, view_sines
    )
    relative_l2_error = cp.linalg.norm(reconstruction - reference) / cp.linalg.norm(
        reference
    )
    if (
        not bool(cp.isfinite(reconstruction).all())
        or not bool(cp.isfinite(relative_l2_error))
        or float(relative_l2_error) >= 1e-4
    ):
        raise RuntimeError(
            f"fused backprojection relative L2 error {float(relative_l2_error):.6g} exceeds tolerance 0.0001"
        )
    print(
        f"fused Tile ramp-filtered reconstruction relative L2 error: {float(relative_l2_error):.6g}"
    )
    print(f"fused Tile voxel-driven = {fused_ms:.3f} ms")
    if arguments.plot:
        save_plot(projections, reconstruction, arguments.plot)


if __name__ == "__main__":
    main()
