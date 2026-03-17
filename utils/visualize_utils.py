from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from .data_utils import (
    denormalize_actions, 
    )
from .pose_utils import *
from PIL import Image, ImageDraw, ImageFont



def _to_pose_array(raw_list) -> np.ndarray:
    """Convert raw loaded data to a float numpy array of shape (T, D>=3)."""

    pose_rows = []
    for idx, item in enumerate(raw_list):
        action_arr = np.asarray(item["actions"], dtype=float).flatten()
        pose_rows.append(action_arr)
        
    pose_array = np.vstack(pose_rows)

    return pose_array

def process_trajectory_episode(
    episode,
    is_normalized: bool,
    stats: Dict,
    pose_type: str,
    reset_pose: np.ndarray,
) -> np.ndarray:
    """Convert raw episode items to 3D trajectory."""

    pose_array = _to_pose_array(episode)

    if is_normalized:
        if stats is None:
            raise ValueError("Stats (mean/std) must be provided for normalized data")
        max, min = (np.asarray(stats["max"]), np.asarray(stats["min"]))
        pose_array = denormalize_actions(pose_array, min, max)

    if pose_type == "absolute":
        traj = pose_array[:, :3]
    else:
        traj = integrate_relative_poses(pose_array, reset_pose)

    return traj


def set_axes_equal(ax):
    """Make 3D plot axes equal scale for true aspect ratio."""

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    half = 0.5 * max_range
    ax.set_xlim3d([x_mid - half, x_mid + half])
    ax.set_ylim3d([y_mid - half, y_mid + half])
    ax.set_zlim3d([z_mid - half, z_mid + half])


def plot_trajectories(trajs: List[np.ndarray], title: str, output: Path | None):
    """Matplotlib 3D plot retained for compatibility."""

    if not trajs:
        raise ValueError("No trajectories to plot")

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")
    for idx, pts in enumerate(trajs):
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Trajectory {idx} must be (T,3), got {pts.shape}")
        color = cmap(idx % 10)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.8, color=color, alpha=0.9)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="green", s=40)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="red", s=40)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    set_axes_equal(ax)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved trajectory overlay to {output}")
    else:
        plt.show()


def plot_trajectories_viser(trajs: List[np.ndarray], title: str, port: int):
    """Serve centered 3D trajectories with axis ticks showing original coordinates."""

    if not trajs:
        raise ValueError("No trajectories to plot")

    try:
        import viser
    except ImportError as exc:  # pragma: no cover - optional
        raise ImportError(
            "viser is required for web visualization. Install with `pip install viser`."
        ) from exc

    all_pts_raw = np.concatenate(trajs, axis=0)
    xyz_min_raw = all_pts_raw.min(axis=0)
    xyz_max_raw = all_pts_raw.max(axis=0)
    original_center = (xyz_min_raw + xyz_max_raw) / 2.0

    centered_trajs = [pts - original_center for pts in trajs]

    all_pts = np.concatenate(centered_trajs, axis=0)
    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)
    span = float(np.max(xyz_max - xyz_min))
    if span <= 1e-6:
        span = 1.0

    server = viser.ViserServer(host="0.0.0.0", port=port)

    corners = np.array([
        [xyz_min[0], xyz_min[1], xyz_min[2]],
        [xyz_max[0], xyz_min[1], xyz_min[2]],
        [xyz_max[0], xyz_max[1], xyz_min[2]],
        [xyz_min[0], xyz_max[1], xyz_min[2]],
        [xyz_min[0], xyz_min[1], xyz_max[2]],
        [xyz_max[0], xyz_min[1], xyz_max[2]],
        [xyz_max[0], xyz_max[1], xyz_max[2]],
        [xyz_min[0], xyz_max[1], xyz_max[2]],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    bbox_segments = np.array([[corners[i], corners[j]] for i, j in edges])

    server.scene.add_line_segments(
        name="/bounding_box",
        points=bbox_segments,
        colors=(200, 200, 200),
        line_width=1.0,
    )

    origin_of_axes = np.array([xyz_min[0], xyz_min[1], xyz_min[2]])
    axis_length = span * 1.1

    axes_segments = np.array([
        [origin_of_axes, origin_of_axes + np.array([axis_length, 0, 0])],
        [origin_of_axes, origin_of_axes + np.array([0, axis_length, 0])],
        [origin_of_axes, origin_of_axes + np.array([0, 0, axis_length])],
    ])

    axis_colors = np.array([
        [[255, 0, 0], [255, 0, 0]],
        [[0, 255, 0], [0, 255, 0]],
        [[0, 0, 255], [0, 0, 255]],
    ])

    server.scene.add_line_segments(
        name="/axes/lines",
        points=axes_segments,
        colors=axis_colors,
        line_width=3.0,
    )

    server.scene.add_label(name="/axes/label_x", text="X (World)", position=origin_of_axes + np.array([axis_length, 0, 0]))
    server.scene.add_label(name="/axes/label_y", text="Y (World)", position=origin_of_axes + np.array([0, axis_length, 0]))
    server.scene.add_label(name="/axes/label_z", text="Z (World)", position=origin_of_axes + np.array([0, 0, axis_length]))

    num_ticks = 5
    tick_length = span * 0.02
    label_offset = span * 0.05

    for ax_idx in range(3):
        centered_ticks = np.linspace(xyz_min[ax_idx], xyz_max[ax_idx], num_ticks)
        world_ticks = centered_ticks + original_center[ax_idx]

        for i in range(num_ticks):
            c_val = centered_ticks[i]
            w_val = world_ticks[i]

            tick_start = origin_of_axes.copy()
            tick_start[ax_idx] = c_val

            tick_end = tick_start.copy()
            if ax_idx == 0:
                tick_end[1] -= tick_length
            else:
                tick_end[0] -= tick_length

            server.scene.add_line_segments(
                name=f"/axes/ticks/axis_{ax_idx}_{i}",
                points=np.array([[tick_start, tick_end]]),
                colors=(0, 0, 0),
                line_width=1.5,
            )

            label_pos = tick_end.copy()
            if ax_idx == 2:
                label_pos[1] -= label_offset
            else:
                label_pos[ax_idx] -= label_offset

            server.scene.add_label(
                name=f"/axes/tick_labels/axis_{ax_idx}_{i}",
                text=f"{w_val:.2f}",
                position=label_pos,
            )

    server.scene.set_up_direction("+z")
    grid_size = max(2.0, span * 2.0)
    server.scene.add_grid(
        name="/grid_xy",
        width=grid_size,
        height=grid_size,
        plane="xy",
        position=(0.0, 0.0, xyz_min[2]),
        cell_size=grid_size / 20,
        section_size=grid_size / 5,
        cell_color=(220, 220, 220),
        section_color=(180, 180, 180),
    )

    palette = [
        (0.121, 0.466, 0.705), (1.000, 0.498, 0.054), (0.173, 0.627, 0.173),
        (0.839, 0.152, 0.156), (0.580, 0.404, 0.741), (0.549, 0.337, 0.294),
        (0.890, 0.467, 0.761), (0.498, 0.498, 0.498), (0.737, 0.741, 0.133),
        (0.090, 0.745, 0.811),
    ]

    for idx, pts in enumerate(centered_trajs):
        color = palette[idx % len(palette)]

        if pts.shape[0] >= 2:
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            server.scene.add_line_segments(
                name=f"/traj_{idx}",
                points=segments,
                colors=color,
                line_width=4.0,
            )

        marker_radius = span * 0.005
        server.scene.add_icosphere(
            name=f"/start_{idx}", position=pts[0], radius=marker_radius, color=(0, 255, 0)
        )
        server.scene.add_icosphere(
            name=f"/end_{idx}", position=pts[-1], radius=marker_radius, color=(255, 0, 0)
        )

    server.scene.add_label(name="/title", text=title, position=(0.0, 0.0, xyz_max[2] + span * 0.1))

    print(f"[viser] Serving at http://localhost:{port} (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("[viser] Server stopped by user")


def plot_xy_overlay(trajs: List[np.ndarray], title: str, output: Path | None):
    """Top-down XY overlay."""

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")

    for idx, pts in enumerate(trajs):
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Trajectory {idx} must be (T,3), got {pts.shape}")
        color = cmap(idx % 10)
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.9, linewidth=1.6)
        ax.scatter(pts[0, 0], pts[0, 1], color="green", s=35)
        ax.scatter(pts[-1, 0], pts[-1, 1], color="red", s=35)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title + " (XY)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved XY overlay to {output}")
    else:
        plt.show()


def plot_xz_overlay(trajs: List[np.ndarray], title: str, output: Path | None):
    """Front-view XZ overlay."""

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")

    for idx, pts in enumerate(trajs):
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Trajectory {idx} must be (T,3), got {pts.shape}")
        color = cmap(idx % 10)
        ax.plot(pts[:, 0], pts[:, 2], color=color, alpha=0.9, linewidth=1.6)
        ax.scatter(pts[0, 0], pts[0, 2], color="green", s=35)
        ax.scatter(pts[-1, 0], pts[-1, 2], color="red", s=35)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(title + " (XZ)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved XZ overlay to {output}")
    else:
        plt.show()



def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 format for more stable imageio behavior."""

    img = np.array(img)

    if img.dtype != np.uint8:
        if img.size > 0 and np.nanmax(img) <= 1.0:
            img = np.clip(img, 0, 1) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame shape and convert to (H, W, 3) uint8."""

    frame = ensure_uint8(frame)

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3:
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        elif frame.shape[-1] != 3:
            raise ValueError(f"Unsupported channel count: {frame.shape[-1]}")
    else:
        raise ValueError(f"Unsupported frame ndim: {frame.ndim}")

    return frame


def overlay_episode_label(frame: np.ndarray, episode_index: int) -> np.ndarray:
    """Draw the episode index at the top-left corner of the frame."""

    if Image is None:
        raise ImportError(
            "Pillow is required for overlaying episode labels when using --concat_all."
        )

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"Episode {episode_index:05d}"

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:  # pragma: no cover - compatibility
        text_w, text_h = draw.textsize(text, font=font)

    padding = 6
    rect_w = text_w + padding * 2
    rect_h = text_h + padding * 2

    draw.rectangle((0, 0, rect_w, rect_h), fill=(0, 0, 0))
    draw.text((padding, padding), text, font=font, fill=(255, 255, 255))

    return np.array(img.convert("RGB"))


def create_video(frames, output_path, frame_rate=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with imageio.get_writer(output_path, fps=frame_rate) as writer:
        for f in frames:
            writer.append_data(normalize_frame(f))


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    episodes = []
    current_episode = []

    for entry in data:
        current_episode.append(entry)
        if entry["dones"] == 1:
            episodes.append(current_episode)
            current_episode = []

    if len(current_episode) > 0:
        episodes.append(current_episode)

    return episodes


def make_output_path(output_dir: str, episode_index: int, key: str, ext: str = ".mp4") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"episode_{episode_index:05d}__{key}{ext}"
    return os.path.join(output_dir, filename)


def make_concat_output_path(output_dir: str, key: str, ext: str = ".mp4") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"all_episodes__{key}{ext}"
    return os.path.join(output_dir, filename)


def try_extract_frame(observation: dict, key: str, frame_index: int = 0):
    """Try extracting one frame from observation[key]."""

    if key not in observation:
        return None, f"key '{key}' not found"

    x = observation[key]
    x = np.array(x)

    if x.size == 0:
        return None, f"key '{key}' is empty"

    if x.ndim == 3:
        if x.shape[-1] in (1, 3, 4):
            return x, None
        if frame_index >= x.shape[0]:
            return None, f"frame_index {frame_index} out of range for '{key}'"
        return x[frame_index], None

    if x.ndim == 4:
        if frame_index >= x.shape[0]:
            return None, f"frame_index {frame_index} out of range for '{key}'"
        return x[frame_index], None

    if x.ndim == 2:
        return x, None

    return None, f"key '{key}' has unsupported shape {x.shape} (ndim={x.ndim})"


def select_keys(first_obs: dict, vis_keys: list[str] | None):
    """Select observation keys to visualize, automatically excluding 'state'."""

    all_keys = list(first_obs.keys())

    if vis_keys is None or len(vis_keys) == 0:
        keys = [k for k in all_keys if k != "state"]
    else:
        keys = [k for k in vis_keys if k != "state"]

    return keys


def process_video_episode(episode_index, file_path, output_dir, vis_keys=None, fps=30, frame_index=0):
    episodes = load_data(file_path)

    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"Episode index {episode_index} out of range. Total episodes: {len(episodes)}")

    episode = episodes[episode_index]
    if len(episode) == 0:
        raise ValueError("Selected episode is empty.")

    first_obs = episode[0]["observations"]
    keys_to_vis = select_keys(first_obs, vis_keys)

    print(f"[Info] Episode {episode_index} length: {len(episode)}")
    print(f"[Info] Keys to visualize (excluding 'state'): {keys_to_vis}")

    for key in keys_to_vis:
        frames = []
        skipped = 0

        for data in episode:
            frame, err = try_extract_frame(data["observations"], key, frame_index=frame_index)
            if frame is None:
                skipped += 1
                continue

            try:
                frames.append(normalize_frame(frame))
            except Exception:
                skipped += 1
                continue

        if len(frames) == 0:
            print(f"[Skip] key='{key}' no valid frames (skipped {skipped}/{len(episode)})")
            continue

        out_path = make_output_path(output_dir, episode_index, key, ext=".mp4")
        create_video(frames, out_path, frame_rate=fps)
        print(f"[OK] key='{key}' saved: {out_path} (frames={len(frames)}, skipped={skipped})")


def process_all_episodes_concat(file_path, output_dir, vis_keys=None, fps=30, frame_index=0):
    episodes = load_data(file_path)

    if len(episodes) == 0:
        raise ValueError("No episodes found in the provided file.")

    first_obs = episodes[0][0]["observations"]
    keys_to_vis = select_keys(first_obs, vis_keys)

    print(f"[Info] Total episodes: {len(episodes)}")
    print(f"[Info] Keys to visualize (excluding 'state'): {keys_to_vis}")

    for key in keys_to_vis:
        combined_frames = []
        skipped = 0

        for ep_idx, episode in enumerate(episodes):
            if len(episode) == 0:
                continue

            for data in episode:
                frame, err = try_extract_frame(data["observations"], key, frame_index=frame_index)
                if frame is None:
                    skipped += 1
                    continue

                try:
                    normalized = normalize_frame(frame)
                    labeled = overlay_episode_label(normalized, ep_idx)
                except ImportError:
                    raise
                except Exception:
                    skipped += 1
                    continue

                combined_frames.append(labeled)

        if len(combined_frames) == 0:
            print(f"[Skip] key='{key}' no valid frames across episodes (skipped {skipped})")
            continue

        out_path = make_concat_output_path(output_dir, key, ext=".mp4")
        create_video(combined_frames, out_path, frame_rate=fps)
        print(f"[OK] key='{key}' concatenated: {out_path} (frames={len(combined_frames)}, skipped={skipped})")

