#!/usr/bin/env python3
"""
Filter successful episodes and save them to a new pkl
(optionally sample successful/failed episodes separately).
Logic:
1. Load the original pkl
2. Split into episodes by dones
3. Mark an episode as successful if the last reward >= 0
4. Merge selected successful episodes (and optional sampled failed episodes) and save
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import *



def filter_and_save(input_path, output_path, sample_episode=None, sample_success=None, sample_failure=None):
    # Load data
    data = load_data(input_path)
    
    # Split episodes
    episodes = split_into_episodes_by_done(data)
    total_episodes = len(episodes)
    print(f"Total episodes found: {total_episodes}")

    # Filter successful episodes
    success_episodes = []
    failed_episodes = []

    for i, ep in enumerate(episodes):
        if not ep: continue
        
        # Get the last transition
        last_step = ep[-1]
        reward = last_step['rewards']
        
        # Criterion: reward >= 0 means success
        if reward >= 0:
            success_episodes.append(ep)
        else:
            failed_episodes.append(ep)

    success_count = len(success_episodes)
    failed_count = len(failed_episodes)
    print(f"-" * 30)
    print(f"Filter Results:")
    print(f"  Success: {success_count} ({(success_count/total_episodes)*100:.1f}%)")
    print(f"  Failed : {failed_count} ({(failed_count/total_episodes)*100:.1f}%)")
    print(f"-" * 30)

    rng = np.random.default_rng()

    # Backward compatibility: sample_episode maps to sampling successful episodes
    if sample_episode is not None and sample_success is None:
        sample_success = sample_episode

    sampled_success = sample_episode_subset(success_episodes, sample_success, "successful", rng)
    if sampled_success is None:
        return

    if sample_failure is None:
        sampled_failure = []
    else:
        sampled_failure = sample_episode_subset(failed_episodes, sample_failure, "failed", rng)
        if sampled_failure is None:
            return

    if not sampled_success and not sampled_failure:
        print("Warning: No episodes selected! Nothing will be saved.")
        return

    # Flatten data
    output_data = []
    for ep in sampled_success + sampled_failure:
        for transition in ep:
            output_data.append(transition)

    print(f"Flattened {len(sampled_success)} successful and {len(sampled_failure)} failed episodes into {len(output_data)} transitions.")

    # Save
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_p, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Successfully saved to: {output_p}")

def parse_args():
    parser = argparse.ArgumentParser(description="Filter successful episodes (optionally sample failures) from pkl dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pkl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pkl file")
    parser.add_argument(
        "--sample_episode",
        type=int,
        default=None,
        help="Optional: randomly sample N successful episodes before saving",
    )
    parser.add_argument(
        "--sample_success",
        type=int,
        default=None,
        help="Optional: randomly sample N successful episodes before saving",
    )
    parser.add_argument(
        "--sample_failure",
        type=int,
        default=None,
        help="Optional: randomly sample N failed episodes before saving (kept only if specified)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    filter_and_save(
        args.input,
        args.output,
        sample_episode=args.sample_episode,
        sample_success=args.sample_success,
        sample_failure=args.sample_failure,
    )
