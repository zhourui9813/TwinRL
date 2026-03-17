import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import jax

def load_data(file_path):
    """Load a pkl file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} transitions")
    print(f"Action Dimention: {data[0]['actions'].shape}")
    return data

def save_pkl(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} transitions -> {path}")

def save_json(path: Path, content):
    print(f"\nSaving statistics to: {path}")
    with path.open("w") as f:
        json.dump(content, f, indent=2)


def find_cutoff_index_by_done(data, target_episodes):
    """
    Scan data from the beginning and find the index where the
    target_episodes-th dones == True appears.
    Returns: (cutoff_idx, actual_episodes)
      - cutoff_idx: The max index to keep (inclusive)
      - actual_episodes: Actual number of episodes found (may be < target_episodes)
    """
    episode_count = 0
    cutoff_idx = len(data) - 1  # Keep all if target_episodes is not reached

    for idx, transition in enumerate(data):
        if transition['dones']:
            episode_count += 1
            if episode_count == target_episodes:
                cutoff_idx = idx
                break

    return cutoff_idx, episode_count


def split_data_at_episode(data, split_episode):
    """
    Split data into two parts by episode index without breaking episodes.

    Args:
      - data: List of transitions, each containing a 'dones' key.
      - split_episode: Episode index to split at (1-based, inclusive).

    Returns:
      (head, tail, actual_episodes)
        - head: Transitions from the first split_episode episodes;
          all data if total episodes are fewer.
        - tail: Remaining transitions; empty if split_episode is not reached.
        - actual_episodes: Actual number of episodes in data.
    """
    if split_episode < 1:
        raise ValueError("split_episode must be >= 1")

    cutoff_idx, actual_eps = find_cutoff_index_by_done(data, split_episode)

    head = data[: cutoff_idx + 1]
    tail = data[cutoff_idx + 1 :] if actual_eps >= split_episode else []

    return head, tail, actual_eps

def split_into_episodes_by_done(data: List[dict]) -> List[List[dict]]:
    """
    Split episodes by transition['dones'].
    Rule: an episode ends when dones == True (that transition is included).
    If the last episode has no done=True, it is kept as an unfinished episode.
    """
    episodes: List[List[dict]] = []
    current: List[dict] = []

    for transition in data:
        current.append(transition)
        if transition.get('dones', False):
            episodes.append(current)
            current = []

    if current:
        episodes.append(current)

    return episodes

def delete_episodes(episodes: List[List[dict]], delete_indices: List[int]):
    """
    Delete specified episodes (0-based).
    Returns: kept transitions (flattened).
    """
    n = len(episodes)

    # Keep only indices within valid range
    valid = [i for i in delete_indices if 0 <= i < n]
    invalid = [i for i in delete_indices if i < 0 or i >= n]

    if invalid:
        print(f"Warning: invalid episode indices (out of range 0..{n-1}): {invalid} (ignored)")

    delete_set = set(valid)

    kept = []
    for i, ep in enumerate(episodes):
        if i in delete_set:
            continue
        else:
            kept.append(ep)
    kept = flatten_episodes_list(kept)
    return kept

def flatten_episodes_list(episodes: List[List[dict]]) -> List[dict]:
    """Flatten an episode list back to a transition list (keep original order)."""
    out: List[dict] = []
    for ep in episodes:
        out.extend(ep)
    return out


def denormalize_actions(normalized_actions: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    return (normalized_actions + 1) * (max_vals - min_vals) / 2 + min_vals


def load_stats(path: Path):
    print(f"Loading statistics: {path}")
    with path.open("r") as f:
        stats = json.load(f)
    return stats


def denormalize_dataset(data, min_vals: np.ndarray, max_vals: np.ndarray):
    print("\nDenormalizing actions...")
    denormalized_data = []
    for _, sample in tqdm(enumerate(data), desc="Denormalizing data", total=len(data)):
        denormalized_sample = sample.copy()
        denormalized_sample["actions"] = denormalize_actions(
            sample["actions"], min_vals, max_vals
        )
        denormalized_data.append(denormalized_sample)
    return denormalized_data



def normalize_actions(actions, min_vals, max_vals, eps=1e-10):
    actions = np.asarray(actions, dtype=np.float32)
    min_vals = np.asarray(min_vals, dtype=np.float32)
    max_vals = np.asarray(max_vals, dtype=np.float32)

    denom = max_vals - min_vals
    zero_mask = np.abs(denom) < eps

    if np.any(zero_mask):
        safe_denom = denom.copy()
        safe_denom[zero_mask] = 1.0

        normalized = 2.0 * (actions - min_vals) / safe_denom - 1.0
        normalized[..., zero_mask] = 1.0
        return normalized
    else:
        return 2.0 * (actions - min_vals) / denom - 1.0

def normalize_dataset(data, min_vals: np.ndarray, max_vals: np.ndarray):
    print("\nNormalizing actions...")
    normalized_data = []
    for _, sample in tqdm(enumerate(data), desc="Normalizing data", total=len(data)):
        normalized_sample = sample.copy()
        normalized_sample["actions"] = normalize_actions(sample["actions"], min_vals, max_vals)
        normalized_data.append(normalized_sample)

    normalized_actions = np.array([sample["actions"] for sample in normalized_data])
    print("\nAction range after normalization:")
    print(f"Min: {normalized_actions.min(axis=0)}")
    print(f"Max: {normalized_actions.max(axis=0)}")
    return normalized_data

def collect_actions(data):
    all_actions = []
    for sample in tqdm(data, desc="Collecting actions"):
        all_actions.append(sample["actions"])
    actions_array = np.array(all_actions)
    print(f"Shape of all actions: {actions_array.shape}")
    return actions_array


def compute_action_stats(actions_array: np.ndarray):
    print("\nComputing statistics...")
    stats = {
        "min": actions_array.min(axis=0).tolist(),
        "max": actions_array.max(axis=0).tolist(),
        "mean": actions_array.mean(axis=0).tolist(),
        "std": actions_array.std(axis=0).tolist(),
        "median": np.median(actions_array, axis=0).tolist(),
        "q01": np.percentile(actions_array, 1, axis=0).tolist(),
        "q99": np.percentile(actions_array, 99, axis=0).tolist(),
    }
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    print("\nAction statistics by dimension:")
    for idx, name in enumerate(dim_names):
        print(
            f"{name}: min={stats['min'][idx]:.4f}, max={stats['max'][idx]:.4f}, "
            f"mean={stats['mean'][idx]:.4f}, std={stats['std'][idx]:.4f}"
        )
    return stats


def sample_episode_subset(pool, k, label, rng):
    """Sample k episodes from pool without replacement; k=None keeps all."""
    if k is None:
        return pool

    if k < 0:
        print(f"{label} must be a non-negative integer. Nothing will be saved.")
        return None

    if k == 0:
        print(f"Skipping {label} episodes because sample size is 0.")
        return []

    total = len(pool)
    if total == 0:
        print(f"No {label} episodes available to sample.")
        return []

    if k < total:
        indices = rng.choice(total, size=k, replace=False)
        sampled = [pool[i] for i in indices]
        print(f"Sampling {k} {label} episodes out of {total}.")
        return sampled

    print(f"Requested {label} sample {k} >= available {total}; keeping all.")
    return pool

def compute_delta_rpy(curr_rpy, next_rpy):
    # curr_quat, next_quat: [x, y, z, w]
    R_curr = R.from_euler("xyz", curr_rpy)
    R_next = R.from_euler("xyz", next_rpy)

    # Desired relation: R_delta * R_curr = R_next
    R_delta = R_next * R_curr.inv()

    # Convert to xyz Euler angles (consistent with env)
    delta_rpy = R_delta.as_euler("xyz", degrees=False)

    # Optional: wrap angles to [-pi, pi]
    delta_rpy = (delta_rpy + np.pi) % (2 * np.pi) - np.pi
    return delta_rpy



def calc_return_to_go(rewards, terminals, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    A config dict for getting the default high/low rewrd values for each envs
    """
    if len(rewards) == 0:
        return np.array([])

    if is_sparse_reward:
        reward_neg = reward_neg * reward_scale + reward_bias
    else:
        assert not is_sparse_reward, "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1-gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i-1] = rewards[-i-1] + gamma * \
                prev_return * (1 - terminals[-i-1])
            prev_return = return_to_go[-i-1]

    return np.array(return_to_go, dtype=np.float32)


def add_mc_returns_to_trajectory(trajectory, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    undate every transition in the trajectory and add mc_returns
    return the updated trajectory
    """
    rewards = [t['rewards'] for t in trajectory]
    terminals = [t['dones'] for t in trajectory]

    mc_returns = calc_return_to_go(
        rewards=rewards,
        terminals=terminals,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=is_sparse_reward,
    )

    for i, transition in enumerate(trajectory):
        transition['mc_returns'] = mc_returns[i]

    return trajectory


def add_embeddings_to_trajectory(trajectory, model, tasks):
    """
    undate every transition in the trajectory and add embeddings
    return the updated trajectory
    """
    for i in range(len(trajectory)):
        observation = trajectory[i]['observations']

        image_primary = observation["side_policy_256"]
        image_wrist = observation["wrist_1"]
        # Add batch dimension
        image_primary = image_primary[np.newaxis, ...]
        image_wrist = image_wrist[np.newaxis, ...]
        timestep_pad_mask = np.array([[True, True]])

        observation = {"image_primary": image_primary,
                       "image_wrist": image_wrist,
                       "timestep_pad_mask": timestep_pad_mask,
                       }

        action_embeddings = model.sample_transformer(observation, tasks,)
        # Now, action_embeddings is (batch_size, window_size, embedding_size)

        # remove window_size dimension
        action_embeddings = action_embeddings[:, -1, :]

        trajectory[i]['embeddings'] = action_embeddings

    return trajectory


def add_next_embeddings_to_trajectory(trajectory):
    """
    undate every transition in the trajectory and add next_embeddings
    return the updated trajectory
    """
    for i in range(len(trajectory)):
        if i == len(trajectory) - 1:
            trajectory[i]['next_embeddings'] = trajectory[i]['embeddings']
        else:
            trajectory[i]['next_embeddings'] = trajectory[i+1]['embeddings']

    return trajectory
