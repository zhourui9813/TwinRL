#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute action statistics and normalized dataset from a pickle file."
    )
    parser.add_argument("--input_path", help="Path to the input pickle file that stores demo transitions.")
    parser.add_argument(
        "--output_stats_path",
        help="Optional JSON file to save the action statistics. Defaults to <input>_action_stats.json.",
    )
    parser.add_argument(
        "--output_normalized_path",
        help="Optional pickle file to save normalized transitions. Defaults to <input>_normalized.pkl.",
    )
    return parser.parse_args()


def derive_default_paths(input_path: Path, stats_path: str | None, normalized_path: str | None):
    if stats_path is None:
        stats_path = input_path.with_name(f"{input_path.stem}_action_stats.json")
    else:
        stats_path = Path(stats_path)

    if normalized_path is None:
        normalized_path = input_path.with_name(f"{input_path.stem}_normalized.pkl")
    else:
        normalized_path = Path(normalized_path)

    return stats_path, normalized_path
    


def main():
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    stats_path, normalized_path = derive_default_paths(input_path, args.output_stats_path, args.output_normalized_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = load_data(input_path)
    all_actions = collect_actions(data)
    action_stats = compute_action_stats(all_actions)
    save_json(stats_path, action_stats)

    min_vals = np.array(action_stats["min"])
    max_vals = np.array(action_stats["max"])

    normalized_data = normalize_dataset(data, min_vals, max_vals)
    save_pkl(normalized_data, normalized_path, )

    print("\nDone!")
    print(f"1. Statistics saved to: {stats_path}")
    print(f"2. Normalized data saved to: {normalized_path}")
    print(f"3. Original data length: {len(data)}")
    print(f"4. Normalized data length: {len(normalized_data)}")


if __name__ == "__main__":
    main()
