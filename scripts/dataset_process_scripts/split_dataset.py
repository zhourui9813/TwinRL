#!/usr/bin/env python3
"""
Split a transition pkl into two parts by episode index using find_cutoff_index_by_done.
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import load_data, find_cutoff_index_by_done, save_pkl

def main():
    parser = argparse.ArgumentParser(description="Split dataset by episode boundary")
    parser.add_argument("--input", required=True, help="Input pkl path")
    parser.add_argument("--split-episode", type=int, required=True, help="Episode index to split at (1-based, inclusive)")
    parser.add_argument("--head", help="Output path for head part (default: <input>_head.pkl)")
    parser.add_argument("--tail", help="Output path for tail part (default: <input>_tail.pkl)")
    args = parser.parse_args()

    data = load_data(args.input)
    cutoff_idx, actual_eps = find_cutoff_index_by_done(data, args.split_episode)
    print(f"Found {actual_eps} episodes; cutoff index = {cutoff_idx}")

    head = data[: cutoff_idx + 1]
    tail = data[cutoff_idx + 1 :] if actual_eps >= args.split_episode else []

    base = Path(args.input)
    head_path = Path(args.head) if args.head else base.with_suffix(base.suffix + "_head.pkl")
    tail_path = Path(args.tail) if args.tail else base.with_suffix(base.suffix + "_tail.pkl")

    save_pkl(head, head_path)
    if tail:
        save_pkl(tail, tail_path)
    else:
        print("Tail part is empty (episodes fewer than split point); nothing saved for tail.")


if __name__ == "__main__":
    main()
