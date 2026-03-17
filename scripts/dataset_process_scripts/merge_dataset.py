#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import (
    load_data,
    save_pkl,
    denormalize_actions,
    load_stats,
    denormalize_dataset
)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge two pickle datasets.")
    parser.add_argument("--file1", required=True, help="Path to first pickle file.")
    parser.add_argument("--file2", required=True, help="Path to second pickle file.")
    parser.add_argument("--output_file", required=True, help="Path to save merged pickle.")
    return parser.parse_args()

def main():
    args = parse_args()
    file1 = Path(args.file1).expanduser().resolve()
    file2 = Path(args.file2).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    print(f"output: {output_file}")

    data1 = load_data(file1)
    data2 = load_data(file2)
    merged_data = data1 + data2

    save_pkl(merged_data, output_file)

    print(f"Merge completed! Total samples: {len(merged_data)}")
    print(f"Dataset 1: {len(data1)} samples")
    print(f"Dataset 2: {len(data2)} samples")
    print(f"After merge: {len(merged_data)} samples")


if __name__ == "__main__":
    main()
