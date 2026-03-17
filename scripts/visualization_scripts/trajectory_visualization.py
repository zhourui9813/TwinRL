from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import *

from utils.visualize_utils import (
    _to_pose_array,
    plot_trajectories,
    plot_trajectories_viser,
    plot_xy_overlay,
    plot_xz_overlay,
    process_trajectory_episode,
)

def parse_reset_pose(reset_pose_str: str | None, dim: int) -> np.ndarray:
    """Parse reset_pose string into numpy array of length >=3."""

    try:
        values = [float(x.strip()) for x in reset_pose_str.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("reset_pose must be comma-separated numbers") from exc

    return np.asarray(values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 3D trajectory overlay from pose/action data")
    parser.add_argument("--data_path", type=Path, help="Path to .npy/.json/.pkl data file")
    parser.add_argument(
        "--pose_type",
        choices=["absolute", "relative"],
        default="absolute",
        help="Interpret data as absolute poses or relative deltas",
    )
    parser.add_argument(
        "--reset_pose",
        type=str,
        default=None,
        help="Initial pose for relative integration (comma separated)",
    )
    parser.add_argument(
        "--is_normalized",
        action="store_true",
        help="Set if the input data is normalized (x = (raw-mean)/std)",
    )
    parser.add_argument(
        "--stat_path",
        type=Path,
        default=None,
        help="Path to stats file containing mean/std (required when normalized)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional base output image path. Saves *_3d.png, *_xy.png, *_xz.png",
    )
    parser.add_argument(
        "--vis_port",
        type=int,
        default=8080,
        help="viser server port for 3D web visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw = load_data(args.data_path)
    episodes = split_into_episodes_by_done(raw)
    if not episodes:
        raise ValueError("No data found in dataset")

    # Inspect first episode to infer dimension for reset pose and stat sanity.
    first_pose_array = _to_pose_array(episodes[0])
    dim = first_pose_array.shape[1]

    stats = None
    if args.is_normalized:
        if args.stat_path is None:
            raise ValueError("--stat_path is required when --is_normalized is set")
        stats = load_stats(args.stat_path)
        mean, std = (np.asarray(stats["mean"]), np.asarray(stats["std"]))
        if mean.shape[-1] != dim or std.shape[-1] != dim:
            raise ValueError(
                f"Mean/std dimension mismatch: stats dim {mean.shape[-1]} vs data dim {dim}"
            )

    reset_pose = parse_reset_pose(args.reset_pose, dim)

    trajectories: List = []
    for ep in episodes:
        traj = process_trajectory_episode(ep, args.is_normalized, stats, args.pose_type, reset_pose)
        trajectories.append(traj)

    title = f"Trajectories ({args.pose_type}) — {len(trajectories)} episode(s)"

    # Derive output paths
    out3d = outxy = outxz = None
    if args.output:
        base = args.output.with_suffix("")
        out3d = base.with_name(base.name + "_3d.png")
        outxy = base.with_name(base.name + "_xy.png")
        outxz = base.with_name(base.name + "_xz.png")

    # 2D overlays (and optional static 3D snapshot) only when output path is provided
    if out3d:
        plot_trajectories(trajectories, title=title, output=out3d)
    if outxy:
        plot_xy_overlay(trajectories, title=title, output=outxy)
    if outxz:
        plot_xz_overlay(trajectories, title=title, output=outxz)

    # 3D via viser web viewer (blocking)
    plot_trajectories_viser(trajectories, title=title, port=args.vis_port)


if __name__ == "__main__":
    main()
