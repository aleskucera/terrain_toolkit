from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def _rasterize_max(
    points: wp.array(dtype=wp.vec3),
    xmin: float,
    ymin: float,
    inv_res: float,
    width: int,
    height: int,
    heightmap: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    p = points[tid]
    j = int((p[0] - xmin) * inv_res)
    i = int((p[1] - ymin) * inv_res)
    if i < 0 or i >= height or j < 0 or j >= width:
        return
    wp.atomic_max(heightmap, i, j, p[2])


@wp.kernel
def _rasterize_sum_count(
    points: wp.array(dtype=wp.vec3),
    xmin: float,
    ymin: float,
    inv_res: float,
    width: int,
    height: int,
    sum_map: wp.array2d(dtype=wp.float32),
    count_map: wp.array2d(dtype=wp.int32),
):
    tid = wp.tid()
    p = points[tid]
    j = int((p[0] - xmin) * inv_res)
    i = int((p[1] - ymin) * inv_res)
    if i < 0 or i >= height or j < 0 or j >= width:
        return
    wp.atomic_add(sum_map, i, j, p[2])
    wp.atomic_add(count_map, i, j, 1)


@dataclass
class HeightMapBuilder:
    resolution: float
    bounds: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    reduction: str = "max"
    fill_value: float = float("nan")

    def __post_init__(self) -> None:
        if self.reduction not in ("max", "mean"):
            raise ValueError(f"Unknown reduction: {self.reduction!r}")
        xmin, xmax, ymin, ymax = self.bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid bounds.")
        self.width = int(math.ceil((xmax - xmin) / self.resolution))
        self.height = int(math.ceil((ymax - ymin) / self.resolution))

    def build(self, points: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")

        xmin, _, ymin, _ = self.bounds
        pts = np.ascontiguousarray(points, dtype=np.float32)
        pts_wp = wp.array(pts, dtype=wp.vec3)
        shape = (self.height, self.width)
        common = [
            pts_wp,
            float(xmin),
            float(ymin),
            float(1.0 / self.resolution),
            int(self.width),
            int(self.height),
        ]

        if self.reduction == "max":
            hm = wp.array(np.full(shape, -np.inf, dtype=np.float32), dtype=wp.float32)
            wp.launch(_rasterize_max, dim=pts.shape[0], inputs=common + [hm])
            wp.synchronize()
            out = hm.numpy().copy()
            out[~np.isfinite(out)] = self.fill_value
            return out

        # mean
        sum_map = wp.zeros(shape, dtype=wp.float32)
        count_map = wp.zeros(shape, dtype=wp.int32)
        wp.launch(
            _rasterize_sum_count,
            dim=pts.shape[0],
            inputs=common + [sum_map, count_map],
        )
        wp.synchronize()
        sums = sum_map.numpy()
        counts = count_map.numpy()
        out = np.full(shape, self.fill_value, dtype=np.float32)
        mask = counts > 0
        out[mask] = sums[mask] / counts[mask]
        return out
