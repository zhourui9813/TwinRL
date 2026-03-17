#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Set observations['state'] and next_observations['state'] to zero arrays."
    )
    parser.add_argument("--use_state", action="store_true", help="keep state")
    parser.add_argument("--input_path",  help="Path to the input pickle file containing transitions.")
    parser.add_argument(
        "--output_path",
        help="Optional output pickle path. Defaults to <input>_zero_state.pkl in the same directory.",
    )
    return parser.parse_args()


def derive_output_path(input_path: Path, output_path: str | None, use_state) -> Path:
    if output_path is None:
        if use_state:
            return input_path.with_name(f"{input_path.stem}_eef_state.pkl")
        else:
            return input_path.with_name(f"{input_path.stem}_zero_state.pkl")
    return Path(output_path)


def zero_states(data, use_state):
    for sample in data:
        
        if "state" in sample["observations"]:
            if not use_state:
                shape = sample["observations"]["state"].shape
                sample["observations"]["state"] = np.zeros(shape, dtype=float)


        if "state" in sample["next_observations"]:
            if not use_state:
                shape2 = sample["next_observations"]["state"].shape
                sample["next_observations"]["state"] = np.zeros(shape2, dtype=float)


def verify_output(path: Path, use_state):
    print(f"Loading: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)

    print(f"Total transitions: {len(data)}")
    nonzero_count = 0
    nonzero_samples = []

    for i, sample in enumerate(data):
        
        obs_state = sample["observations"]["state"]
        next_state = sample["next_observations"]["state"]
        print("obs_state.shape: ", obs_state.shape, "obs_state: ", obs_state)
        print("next_state.shape: ", next_state.shape, "next_state: ", next_state)
        # import pdb;pdb.set_trace()

        if not np.allclose(obs_state, 0):
            nonzero_count += 1
            nonzero_samples.append(("obs", i, obs_state.copy()))

        if not np.allclose(next_state, 0):
            nonzero_count += 1
            nonzero_samples.append(("next_obs", i, next_state.copy()))

        # if len(nonzero_samples) >= 20:
        #     break

    print("\n===== Verification Results =====")
    if not use_state:
        if nonzero_count == 0:
            print("All state and next_state values are zero.")
        else:
            print(f"Found {nonzero_count} non-zero states.")
            print("Below are up to the first 20 non-zero samples:\n")
            for tag, idx, st in nonzero_samples:
                print(f" - [{tag}] Non-zero at sample index={idx}, state shape={st.shape}")
                nz = np.argwhere(st != 0)
                print(f"   Non-zero positions (showing up to 10): {nz[:10].tolist()}")
                print(f"   Values (showing up to 10): {st[st != 0][:10]}")
                print(" ")
    else:
        print(f"None zero count {nonzero_count}")
        print(f"Len data {len(data)}")
        if nonzero_count == 2 * len(data):
            print("Correct state saved")
        else:
            print("Wrong state saved")


def main():
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = derive_output_path(input_path, args.output_path, args.use_state)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("rb") as f:
        data = pickle.load(f)

    zero_states(data, args.use_state)

    with output_path.open("wb") as f:
        pickle.dump(data, f)

    print("Done! All states set to zero.")
    verify_output(output_path, args.use_state)


if __name__ == "__main__":
    main()
