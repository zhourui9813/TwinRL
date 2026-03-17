import pickle
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import *

def derive_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path is None or output_path == '':
        data_dir = os.path.dirname(input_path)
        base_name = os.path.basename(input_path).split('.')[0]
        output_path = os.path.join(data_dir, f'{base_name}_delta_ee.pkl')
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Set observations['state'] and next_observations['state'] to zero arrays."
    )
    parser.add_argument("--input_path", help="Path to the input pickle file containing transitions.")
    parser.add_argument(
        "--output_path",
        help="Optional output pickle path. Defaults to <input>_zero_state.pkl in the same directory.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = derive_output_path(input_path, args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "rb") as fin:
        data = pickle.load(fin)

    diffs = []
    done_indices = []
    big_y_indices = []
    big_z_indices = []

    for i in range(len(data)):
        action = data[i]["actions"]
        action_i = action
        euler_angles_i = action_i[3:6]
        done = data[i]['dones']
        if done:
            done_indices.append(i)
            # diffs.append(np.zeros(7))

        if i==0:
            action_diff = np.zeros(action.shape)
            action_diff[-1] = action_i[-1]
            diffs.append(action_diff)

        if i > 0:
            action_diff_pos = action_i[:3] - action_i_prev[:3]
            action_diff_rpy = compute_delta_rpy(euler_angles_i_prev, euler_angles_i)
            action_diff = np.concatenate([action_diff_pos, action_diff_rpy, action_i[-1:]])
            if done_indices and i==done_indices[-1]+1:
                action_diff = np.zeros(action.shape)
                action_diff[-1] = action_i[-1]
                # print('action_diff:', action_diff)
            diffs.append(action_diff)
            if action_diff[1] > 0.1:
                big_y_indices.append(i)
            if action_diff[2] > 0.1:
                big_z_indices.append(i)
        action_i_prev = action_i
        euler_angles_i_prev = euler_angles_i

    diffs = np.array(diffs)

    print("=== action_diff range (per dimension) ===")
    print("min:", diffs.min(axis=0))
    print("max:", diffs.max(axis=0))
    print("mean:", diffs.mean(axis=0))
    print("std:", diffs.std(axis=0))

    print("done_indices:", done_indices)
    print("big_y_indices:", big_y_indices)
    print("big_z_indices:", big_z_indices)

    print('len(diffs):', len(diffs))
    print('len(done_indices):', len(done_indices))
    print('len(big_y_indices):', len(big_y_indices))
    print('len(big_z_indices):', len(big_z_indices))
    print('len(data):', len(data))

    from tqdm import tqdm
    delta_ee_data = []
    for i, sample in tqdm(enumerate(data), desc="change to delta_ee", total=len(data)):
        delta_ee_sample = sample.copy()
        delta_ee_sample['actions'] = diffs[i]
        delta_ee_data.append(delta_ee_sample)

    print('len(delta_ee_data):', len(delta_ee_data))

    with open(output_path, 'wb') as f:
        pickle.dump(delta_ee_data, f)

    print("save to:", output_path)


if __name__ == "__main__":
    main()
