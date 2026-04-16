from __future__ import annotations

import math

import numpy as np
import warp as wp


@wp.kernel
def _blur_axis(
    src: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    radius: int,
    axis: int,  # 0 = vertical (along i), 1 = horizontal (along j)
    dst: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    h = src.shape[0]
    w = src.shape[1]
    acc = float(0.0)
    wsum = float(0.0)
    for k in range(-radius, radius + 1):
        ii = i
        jj = j
        if axis == 0:
            ii = i + k
        else:
            jj = j + k
        if ii < 0 or ii >= h or jj < 0 or jj >= w:
            continue
        v = src[ii, jj]
        if wp.isnan(v):
            continue
        wgt = weights[k + radius]
        acc += v * wgt
        wsum += wgt
    if wsum > 0.0:
        dst[i, j] = acc / wsum
    else:
        dst[i, j] = wp.float32(wp.nan)


@wp.kernel
def _diffuse_step(
    src: wp.array2d(dtype=wp.float32),
    fixed: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    if fixed[i, j] == 1:
        dst[i, j] = src[i, j]
        return
    h = src.shape[0]
    w = src.shape[1]
    acc = float(0.0)
    count = float(0.0)
    if i > 0:
        v = src[i - 1, j]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if i < h - 1:
        v = src[i + 1, j]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if j > 0:
        v = src[i, j - 1]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if j < w - 1:
        v = src[i, j + 1]
        if not wp.isnan(v):
            acc += v
            count += 1.0
    if count > 0.0:
        dst[i, j] = acc / count
    else:
        dst[i, j] = wp.float32(wp.nan)


@wp.kernel
def _downsample(
    src: wp.array2d(dtype=wp.float32),
    src_fixed: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.float32),
    dst_fixed: wp.array2d(dtype=wp.int32),
):
    """2x2 average downsample, NaN-aware. A cell is fixed if any source cell was fixed."""
    i, j = wp.tid()
    si = i * 2
    sj = j * 2
    sh = src.shape[0]
    sw = src.shape[1]
    acc = float(0.0)
    count = float(0.0)
    any_fixed = int(0)
    for di in range(2):
        for dj in range(2):
            ii = si + di
            jj = sj + dj
            if ii < sh and jj < sw:
                v = src[ii, jj]
                if not wp.isnan(v):
                    acc += v
                    count += 1.0
                if src_fixed[ii, jj] == 1:
                    any_fixed = 1
    if count > 0.0:
        dst[i, j] = acc / count
    else:
        dst[i, j] = wp.float32(wp.nan)
    dst_fixed[i, j] = any_fixed


@wp.kernel
def _upsample_inject(
    coarse: wp.array2d(dtype=wp.float32),
    fine: wp.array2d(dtype=wp.float32),
    fine_fixed: wp.array2d(dtype=wp.int32),
):
    """Upsample coarse solution to fine grid; only write into non-fixed cells."""
    i, j = wp.tid()
    if fine_fixed[i, j] == 1:
        return
    ci = i / 2
    cj = j / 2
    if ci < coarse.shape[0] and cj < coarse.shape[1]:
        v = coarse[ci, cj]
        if not wp.isnan(v):
            fine[i, j] = v


def _run_diffusion(a: wp.array2d, b: wp.array2d, fixed: wp.array2d, iters: int) -> wp.array2d:
    """Run `iters` diffusion steps using CUDA graph capture, return buffer with result."""
    if iters <= 0:
        return a
    wp.capture_begin()
    try:
        wp.launch(_diffuse_step, dim=(a.shape[0], a.shape[1]), inputs=[a, fixed, b])
        wp.launch(_diffuse_step, dim=(a.shape[0], a.shape[1]), inputs=[b, fixed, a])
    finally:
        graph = wp.capture_end()
    for _ in range(iters // 2):
        wp.capture_launch(graph)
    if iters % 2:
        wp.launch(_diffuse_step, dim=(a.shape[0], a.shape[1]), inputs=[a, fixed, b])
        a, b = b, a
    return a


def multigrid_inpaint(
    heightmap: np.ndarray, iters_per_level: int = 50, coarse_iters: int = 200, min_size: int = 8,
) -> np.ndarray:
    """Multigrid diffusion inpainting: solve coarse, upsample, refine at each level."""
    if heightmap.ndim != 2:
        raise ValueError("heightmap must be 2D.")

    data = np.ascontiguousarray(heightmap, dtype=np.float32)
    fixed_np = np.isfinite(data).astype(np.int32)

    # Build pyramid (finest → coarsest).
    levels = [(wp.array(data, dtype=wp.float32), wp.array(fixed_np, dtype=wp.int32))]
    while levels[-1][0].shape[0] > min_size and levels[-1][0].shape[1] > min_size:
        prev, prev_f = levels[-1]
        ch = (prev.shape[0] + 1) // 2
        cw = (prev.shape[1] + 1) // 2
        coarse = wp.zeros((ch, cw), dtype=wp.float32)
        coarse_f = wp.zeros((ch, cw), dtype=wp.int32)
        wp.launch(_downsample, dim=(ch, cw), inputs=[prev, prev_f, coarse, coarse_f])
        levels.append((coarse, coarse_f))

    # Solve coarsest level.
    a, fixed = levels[-1]
    b = wp.zeros_like(a)
    a = _run_diffusion(a, b, fixed, coarse_iters)
    levels[-1] = (a, fixed)

    # Upsample and refine from coarse to fine.
    for lvl in range(len(levels) - 2, -1, -1):
        fine, fine_fixed = levels[lvl]
        coarse_solved = levels[lvl + 1][0]
        wp.launch(_upsample_inject, dim=(fine.shape[0], fine.shape[1]),
                  inputs=[coarse_solved, fine, fine_fixed])
        b = wp.zeros_like(fine)
        fine = _run_diffusion(fine, b, fine_fixed, iters_per_level)
        levels[lvl] = (fine, fine_fixed)

    wp.synchronize()
    return levels[0][0].numpy().copy()


def diffuse_inpaint(heightmap: np.ndarray, max_iters: int = 500) -> np.ndarray:
    """Fill NaN cells by iterative diffusion from known cells (Laplace inpainting on GPU).

    Uses CUDA graph capture to replay the iteration loop without per-launch Python overhead.
    """
    if heightmap.ndim != 2:
        raise ValueError("heightmap must be 2D.")

    data = np.ascontiguousarray(heightmap, dtype=np.float32)
    fixed_np = np.isfinite(data).astype(np.int32)

    a = wp.array(data, dtype=wp.float32)
    b = wp.zeros(data.shape, dtype=wp.float32)
    fixed = wp.array(fixed_np, dtype=wp.int32)

    # Capture a CUDA graph for 2 iterations (one ping-pong round).
    wp.capture_begin()
    try:
        wp.launch(_diffuse_step, dim=data.shape, inputs=[a, fixed, b])
        wp.launch(_diffuse_step, dim=data.shape, inputs=[b, fixed, a])
    finally:
        graph = wp.capture_end()

    # Each replay does 2 iterations; result ends up in `a`.
    full_rounds = max_iters // 2
    remainder = max_iters % 2
    for _ in range(full_rounds):
        wp.capture_launch(graph)
    if remainder:
        wp.launch(_diffuse_step, dim=data.shape, inputs=[a, fixed, b])
        a, b = b, a
    wp.synchronize()
    return a.numpy().copy()


def gaussian_smooth(heightmap: np.ndarray, sigma: float = 1.0, truncate: float = 3.0) -> np.ndarray:
    """NaN-aware separable Gaussian blur on a 2D heightmap (sigma in cells)."""
    if sigma <= 0.0:
        return heightmap.copy()
    if heightmap.ndim != 2:
        raise ValueError("heightmap must be 2D.")

    radius = max(1, int(math.ceil(truncate * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    weights = np.exp(-(k * k) / (2.0 * sigma * sigma)).astype(np.float32)

    src = wp.array(np.ascontiguousarray(heightmap, dtype=np.float32), dtype=wp.float32)
    tmp = wp.zeros(heightmap.shape, dtype=wp.float32)
    dst = wp.zeros(heightmap.shape, dtype=wp.float32)
    w_wp = wp.array(weights, dtype=wp.float32)

    wp.launch(_blur_axis, dim=heightmap.shape, inputs=[src, w_wp, radius, 1, tmp])
    wp.launch(_blur_axis, dim=heightmap.shape, inputs=[tmp, w_wp, radius, 0, dst])
    wp.synchronize()
    return dst.numpy().copy()
