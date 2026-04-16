import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from elevation_mapping import HeightMapBuilder, gaussian_smooth

BOUNDS = (-5.0, 5.0, -5.0, 5.0)
RESOLUTION = 0.1


def make_synthetic_cloud(
    n: int = 200_000,
    seed: int = 0,
    noise_std: float = 0.01,
    outlier_frac: float = 0.0,
    outlier_std: float = 0.5,
    dropout_frac: float = 0.0,
) -> np.ndarray:
    """Generate a tilted plane + Gaussian bump point cloud with configurable noise.

    - noise_std: stdev of per-point Gaussian z noise (meters)
    - outlier_frac: fraction of points that get large extra z noise (spikes)
    - outlier_std: stdev of the outlier z perturbation (meters)
    - dropout_frac: fraction of points to remove (simulates sparse cloud)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(BOUNDS[0], BOUNDS[1], n)
    y = rng.uniform(BOUNDS[2], BOUNDS[3], n)
    z = 0.1 * x + 0.05 * y + 1.5 * np.exp(-((x - 1.0) ** 2 + (y + 1.0) ** 2) / 1.5)
    z += rng.normal(0.0, noise_std, n)

    if outlier_frac > 0.0:
        mask = rng.random(n) < outlier_frac
        z[mask] += rng.normal(0.0, outlier_std, mask.sum())

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    if dropout_frac > 0.0:
        keep = rng.random(n) >= dropout_frac
        pts = pts[keep]
    return pts


def grid_axes(hm_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = hm_shape
    xmin, _, ymin, _ = BOUNDS
    x = xmin + (np.arange(w) + 0.5) * RESOLUTION
    y = ymin + (np.arange(h) + 0.5) * RESOLUTION
    return x, y


def parse_args() -> argparse.Namespace:
    presets = {
        "clean": dict(noise_std=0.01, outlier_frac=0.0, dropout_frac=0.0),
        "noisy": dict(noise_std=0.05, outlier_frac=0.0, dropout_frac=0.0),
        "very_noisy": dict(noise_std=0.15, outlier_frac=0.02, outlier_std=0.5, dropout_frac=0.0),
        "sparse": dict(noise_std=0.05, outlier_frac=0.01, outlier_std=0.3, dropout_frac=0.9),
    }
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(presets), default="clean")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=None)
    p.add_argument("--outlier-frac", type=float, default=None)
    p.add_argument("--outlier-std", type=float, default=None)
    p.add_argument("--dropout-frac", type=float, default=None)
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--cloud-max-points", type=int, default=20_000,
                   help="Downsample cap for the raw cloud scatter (0 disables panel).")
    args = p.parse_args()
    # Apply preset, then override with any explicit flags.
    cfg = dict(presets[args.preset])
    for k in ("noise_std", "outlier_frac", "outlier_std", "dropout_frac"):
        v = getattr(args, k.replace("_", "_"))
        if v is not None:
            cfg[k] = v
    args.cfg = cfg
    return args


def main() -> None:
    args = parse_args()
    pts = make_synthetic_cloud(n=args.n, seed=args.seed, **args.cfg)
    print(f"preset={args.preset} cfg={args.cfg} points={len(pts)}")

    builder_max = HeightMapBuilder(RESOLUTION, BOUNDS, reduction="max")
    builder_mean = HeightMapBuilder(RESOLUTION, BOUNDS, reduction="mean")
    hm_max = builder_max.build(pts)
    hm_mean = builder_mean.build(pts)
    hm_smooth = gaussian_smooth(hm_max, sigma=args.smooth_sigma)
    x, y = grid_axes(hm_max.shape)

    show_cloud = args.cloud_max_points > 0
    cols = 4 if show_cloud else 3
    titles = ["Raw cloud"] if show_cloud else []
    titles += ["Max", "Mean", f"Max smoothed (σ={args.smooth_sigma})"]
    fig = make_subplots(
        rows=1,
        cols=cols,
        specs=[[{"type": "scene"}] * cols],
        subplot_titles=titles,
    )

    col = 1
    if show_cloud:
        if len(pts) > args.cloud_max_points:
            idx = np.random.default_rng(0).choice(len(pts), args.cloud_max_points, replace=False)
            sub = pts[idx]
        else:
            sub = pts
        fig.add_trace(
            go.Scatter3d(
                x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
                mode="markers",
                marker=dict(size=1.2, color=sub[:, 2], colorscale="Viridis"),
            ),
            1, col,
        )
        col += 1
    fig.add_trace(go.Surface(x=x, y=y, z=hm_max, colorscale="Viridis", showscale=False), 1, col); col += 1
    fig.add_trace(go.Surface(x=x, y=y, z=hm_mean, colorscale="Viridis", showscale=False), 1, col); col += 1
    fig.add_trace(go.Surface(x=x, y=y, z=hm_smooth, colorscale="Viridis", showscale=False), 1, col)
    xmin, xmax, ymin, ymax = BOUNDS
    zmin = float(np.nanmin([np.nanmin(hm_max), np.nanmin(hm_mean), np.nanmin(hm_smooth), pts[:, 2].min()]))
    zmax = float(np.nanmax([np.nanmax(hm_max), np.nanmax(hm_mean), np.nanmax(hm_smooth), pts[:, 2].max()]))
    scene = dict(
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        zaxis=dict(range=[zmin, zmax]),
        aspectmode="cube",
    )
    fig.update_layout(
        title=f"Heightmap [{args.preset}] ({hm_max.shape[1]}×{hm_max.shape[0]} @ {RESOLUTION} m/cell)",
        height=600,
        scene=scene,
        scene2=scene,
        scene3=scene,
        scene4=scene,
    )
    out = "heightmap.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    print(f"max:  z=[{np.nanmin(hm_max):.3f}, {np.nanmax(hm_max):.3f}]")
    print(f"mean: z=[{np.nanmin(hm_mean):.3f}, {np.nanmax(hm_mean):.3f}]")


if __name__ == "__main__":
    main()
