#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Denormalize actions from a pickle file using statistics JSON."
    )
    parser.add_argument("--input_path", help="Path to the normalized pickle file.")
    parser.add_argument("--stats_path", help="Path to the JSON file that stores action statistics.")
    parser.add_argument(
        "--output_path",
        help="Optional pickle file to save denormalized transitions. Defaults to <input>_denormalized.pkl.",
    )
    return parser.parse_args()


def derive_default_paths(input_path: Path, output_path: str | None):
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_denormalized.pkl")
    else:
        output_path = Path(output_path)
    return output_path


def main():
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    stats_path = Path(args.stats_path).expanduser().resolve()
    output_path = derive_default_paths(input_path, args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    data = load_data(input_path)
    stats = load_stats(stats_path)

    min_vals = np.array(stats["min"], dtype=np.float32)
    max_vals = np.array(stats["max"], dtype=np.float32)

    denormalized_data = denormalize_dataset(data, min_vals, max_vals)
    save_pkl(denormalized_data, output_path)

    print("\nDone!")
    print(f"1. Statistics loaded from: {stats_path}")
    print(f"2. Denormalized data saved to: {output_path}")
    print(f"3. Original data length: {len(data)}")
    print(f"4. Denormalized data length: {len(denormalized_data)}")


if __name__ == "__main__":
    main()
