# Twin-RL `scripts` Usage Guide

This guide details the usage of various data processing, utility, and visualization scripts within the Twin-RL repository.

---

## 1. Data Conversion and Preprocessing

### 1.1 `data_format_conversion.py`

**Path:** `scripts/dataset_process_scripts/data_format_conversion.py`

**Purpose:** Convert raw Twin output data (`*.pkl.gz`) into training transition data, build `observation/action/reward/done` fields, and add Octo embeddings.

**Raw Twin Data Structure:**

```
Raw_Twin_data/
├── 0/
│   └── 0.pkl.gz
├── 1/
│   └── 1.pkl.gz
├── ...
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input-raw` | Yes | Raw data directory; recursively reads `*.pkl.gz` |
| `--output` | Yes | Output `.pkl` path |
| `--config` | Yes | Config file path for the target training task |
| `--octo-model-path` | Yes | Path to the Octo pretrained model |
| `--hand` | No | Arm prefix (e.g., `left`). Default: `left` |
| `--delta-gripper` | No | If set, use gripper delta action between adjacent frames (`g[t+1] - g[t]`) |
| `--fix-gripper` | No | If set, fix the gripper state to a constant over the whole trajectory |
| `--fix-gripper-state` | No | Fixed gripper value (only effective when `--fix-gripper` is set). Default: `1.0` |
| `--stack-obs-num` | No | Number of stacked observation frames. Default: `2` |
| `--action-def` | No | Action definition (`absolute` or `relative`). Default: `absolute` |

**Gripper Processing Logic:**

| Mode | Behavior |
|---|---|
| `--fix-gripper` | All values in `gripper_series` are set to `fix_gripper_state` |
| `--delta-gripper` | Binarize raw gripper values to 0/1, then convert to delta (`-1` / `0` / `1`) |
| Neither flag | Purely binarize to 0/1 |

**Examples:**

```bash
# 1. Fixed Gripper
python data_format_conversion.py \
  --input-raw /path/to/raw_data \
  --output /path/to/out_fix_gripper.pkl \
  --config /path/to/task_config.py \
  --hand left \
  --stack-obs-num 2 \
  --action-def absolute \
  --octo-model-path /path/to/octo-small \
  --fix-gripper \
  --fix-gripper-state 1.0

# 2. Gripper Delta
python data_format_conversion.py \
  --input-raw /path/to/raw_data \
  --output /path/to/out_delta_gripper.pkl \
  --config /path/to/task_config.py \
  --hand left \
  --stack-obs-num 2 \
  --action-def absolute \
  --octo-model-path /path/to/octo-small \
  --delta-gripper

# 3. Default Binarized Gripper State
python data_format_conversion.py \
  --input-raw /path/to/raw_data \
  --output /path/to/out_binarized_gripper.pkl \
  --config /path/to/task_config.py \
  --hand left \
  --stack-obs-num 2 \
  --action-def absolute \
  --octo-model-path /path/to/octo-small
```

---

### 1.2 `offline_train_dataset_preprocess.sh`

**Path:** `scripts/dataset_process_scripts/offline_train_dataset_preprocess.sh`

**Purpose:** Preprocess offline training datasets sequentially, including absolute-to-delta action conversion, normalization, and state preprocessing.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `<INPUT_PKL>` | Yes | Path to the input `.pkl` dataset file |
| `--no_normalize` | No | Skip the normalization step |

**Examples:**

```bash
cd scripts/dataset_process_scripts

# Execute full pipeline (including normalization)
bash offline_train_dataset_preprocess.sh /path/to/dataset.pkl

# Execute pipeline but skip normalization
bash offline_train_dataset_preprocess.sh /path/to/dataset.pkl --no_normalize
```

---

## 2. Visualization Scripts

### 2.1 `trajectory_visualization.py`

**Path:** `scripts/visualization_scripts/trajectory_visualization.py`

**Purpose:** Extract episode trajectories from a dataset, generate 2D/3D trajectory visualizations, and launch a `viser` 3D web viewer.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `data_path` | Yes (positional) | Input data (`.pkl` / `.json` / `.npy`) |
| `--pose_type` | No | `absolute` or `relative`. Default: `absolute` |
| `--reset_pose` | No | Integration start pose for relative trajectories (comma-separated) |
| `--is_normalized` | No | Enable if input data is normalized |
| `--stat_path` | Conditional | Normalization stats file (required when `--is_normalized` is set) |
| `--output` | No | Output image prefix. Generates `*_3d.png`, `*_xy.png`, `*_xz.png` |
| `--vis_port` | No | `viser` web server port. Default: `8080` |

**Example:**

```bash
python trajectory_visualization.py \
  /path/to/data.pkl \
  --output /path/to/traj.png \
  --stat_path /path/to/action_stats.json \
  --is_normalized \
  --pose_type relative \
  --reset_pose "0.48,0.0,0.30,3.14,0.0,0.0" \
  --vis_port 8080
```

> **Note:** 2D trajectory visualizations are saved to the specified output path. For 3D visualization, open the local port shown in the terminal in your browser.

---

### 2.2 `visualize_camera_video.py`

**Path:** `scripts/visualization_scripts/visualize_camera_video.py`

**Purpose:** Extract image observations from a dataset and export either single-episode videos or concatenated all-episode videos.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input_file` | Yes | Input `.pkl` path |
| `--output_dir` | Yes | Output directory path |
| `--episode_index` | Conditional | Target episode index (required in single-episode mode) |
| `--concat_all` | No | Process all episodes and concatenate them |
| `--fps` | No | Video frame rate. Default: `5` |
| `--vis_keys` | No | Observation keys to visualize. Default: all keys except `state` |
| `--frame_index` | No | Target frame index when observation has an extra temporal dimension. Default: `0` |

**Examples:**

```bash
# Concatenate all episodes
python visualize_camera_video.py \
  --input_file /path/to/data.pkl \
  --output_dir /path/to/output_videos \
  --fps 10 \
  --concat_all

# Extract a single episode
python visualize_camera_video.py \
  --input_file /path/to/data.pkl \
  --output_dir /path/to/output_videos \
  --episode_index 50 \
  --fps 10
```

---

## 3. Dataset Utility Scripts

### 3.1 `convert_absolute_to_delta.py`

**Path:** `scripts/dataset_process_scripts/convert_absolute_to_delta.py`

**Purpose:** Convert `actions` in each transition from absolute pose to frame-to-frame delta.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input_path` | Yes | Input `.pkl` path |
| `--output_path` | No | Output `.pkl` path. Default: `<input>_delta_ee.pkl` |

**Example:**

```bash
python convert_absolute_to_delta.py \
  --input_path /path/to/input.pkl \
  --output_path /path/to/output_delta.pkl
```

---

### 3.2 `delete_dataset_episodes.py`

**Path:** `scripts/dataset_process_scripts/delete_dataset_episodes.py`

**Purpose:** Delete selected episodes from a dataset by episode index (split by `dones`).

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input` | Yes | Input `.pkl` path |
| `--indices` | Yes | Episode indices to delete. Supports spaces (`"0 3 5"`) or commas (`"0,3,5"`) |
| `--output` | No | Output `.pkl` path. Default: `<input>.pkl_del.pkl` |

**Example:**

```bash
python delete_dataset_episodes.py \
  --input /path/to/data.pkl \
  --indices "0 1 3 5 8" \
  --output /path/to/data_deleted.pkl
```

---

### 3.3 `normalize_actions.py`

**Path:** `scripts/dataset_process_scripts/normalize_actions.py`

**Purpose:** Compute action `min/max/mean/std` and output a normalized dataset.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input_path` | Yes | Input `.pkl` path |
| `--output_stats_path` | No | Output `.json` path for statistics. Default: `<input>_action_stats.json` |
| `--output_normalized_path` | No | Output `.pkl` path for normalized data. Default: `<input>_normalized.pkl` |

**Example:**

```bash
python normalize_actions.py \
  --input_path /path/to/data.pkl \
  --output_stats_path /path/to/data_action_stats.json \
  --output_normalized_path /path/to/data_normalized.pkl
```

---

### 3.4 `denormalize_actions.py`

**Path:** `scripts/dataset_process_scripts/denormalize_actions.py`

**Purpose:** Restore normalized actions to original scale using a statistics file.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input_path` | Yes | Normalized data `.pkl` path |
| `--stats_path` | Yes | Action statistics `.json` path |
| `--output_path` | No | Output `.pkl` path. Default: `<input>_denormalized.pkl` |

**Example:**

```bash
python denormalize_actions.py \
  --input_path /path/to/data_normalized.pkl \
  --stats_path /path/to/data_action_stats.json \
  --output_path /path/to/data_denormalized.pkl
```

---

### 3.5 `merge_dataset.py`

**Path:** `scripts/dataset_process_scripts/merge_dataset.py`

**Purpose:** Merge two `.pkl` datasets via direct list concatenation.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--file1` | Yes | First `.pkl` path |
| `--file2` | Yes | Second `.pkl` path |
| `--output_file` | Yes | Output merged `.pkl` path |

**Example:**

```bash
python merge_dataset.py \
  --file1 /path/to/data1.pkl \
  --file2 /path/to/data2.pkl \
  --output_file /path/to/merged.pkl
```

---

### 3.6 `sample_sucess_fail_episodes.py`

**Path:** `scripts/dataset_process_scripts/sample_sucess_fail_episodes.py`

**Purpose:** Filter samples based on success (`reward >= 0` at the final step), randomly sample a specified number of successful/failed episodes, then flatten and save.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input` | Yes | Input `.pkl` path |
| `--output` | Yes | Output `.pkl` path |
| `--sample_success` | No | Number of successful episodes to sample randomly |
| `--sample_failure` | No | Number of failed episodes to sample randomly |
| `--sample_episode` | No | Backward-compatible alias for `--sample_success` |

**Example:**

```bash
python sample_sucess_fail_episodes.py \
  --input /path/to/data.pkl \
  --output /path/to/data_success.pkl \
  --sample_success 10 \
  --sample_failure 0
```

---

### 3.7 `split_dataset.py`

**Path:** `scripts/dataset_process_scripts/split_dataset.py`

**Purpose:** Split a dataset into head/tail parts on episode boundaries.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input` | Yes | Input `.pkl` path |
| `--split-episode` | Yes | Episode index for splitting (1-based; included in head split) |
| `--head` | No | Output path for head split. Default: `<input>.pkl_head.pkl` |
| `--tail` | No | Output path for tail split. Default: `<input>.pkl_tail.pkl` |

**Example:**

```bash
python split_dataset.py \
  --input /path/to/data.pkl \
  --split-episode 20 \
  --head /path/to/data_head.pkl \
  --tail /path/to/data_tail.pkl
```

---

### 3.8 `state_process.py`

**Path:** `scripts/dataset_process_scripts/state_process.py`

**Purpose:** Process `observations["state"]` and `next_observations["state"]` within transitions.

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input_path` | Yes | Input `.pkl` path |
| `--output_path` | No | Output `.pkl` path. Default: `<input>_zero_state.pkl` or `<input>_eef_state.pkl` |
| `--use_state` | No | If set, retain the physical state; otherwise, zero out the state array |

**Examples:**

```bash
# Zero out state
python state_process.py \
  --input_path /path/to/data.pkl

# Keep state
python state_process.py \
  --input_path /path/to/data.pkl \
  --use_state
```