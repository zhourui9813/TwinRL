#!/usr/bin/env python3
"""
Delete specified episodes from a transition pkl using utils.data_utils.delete_episodes.
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import (
    load_data,
    split_into_episodes_by_done,
    delete_episodes,
    save_pkl
)


def parse_indices(indices_str: str):
    """
    Parse indices like '0 3 5' or '0,3,5' into sorted unique int list.
    """
    tokens = indices_str.replace(",", " ").split()
    out = []
    bad = []
    for t in tokens:
        try:
            out.append(int(t))
        except ValueError:
            bad.append(t)
    if bad:
        print(f"Warning: skipped non-integer tokens: {bad}")
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(description="Delete episodes from dataset by index (0-based).")
    parser.add_argument("--input", required=True, help="Input pkl path")
    parser.add_argument("--indices", required=True, help="Episode indices to delete, e.g. '0 5 8' or '0,5,8'")
    parser.add_argument("--output", help="Output pkl path (default: <input>_del.pkl)")
    args = parser.parse_args()

    data = load_data(args.input)
    episodes = split_into_episodes_by_done(data)
    print(f"Split into {len(episodes)} episodes")

    delete_list = parse_indices(args.indices)
    kept_data= delete_episodes(episodes, delete_list)

    kept_episodes = split_into_episodes_by_done(kept_data)

    out_path = Path(args.output) if args.output else Path(args.input).with_suffix(Path(args.input).suffix + "_del.pkl")
    save_pkl(kept_data, out_path)
    print(f"Episodes kept: {len(kept_episodes)}, transitions kept: {len(kept_data)}")


if __name__ == "__main__":
    main()
