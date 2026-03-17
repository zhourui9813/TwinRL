
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gzip
import glob
import pickle
import argparse
import importlib.util
import sys
from collections import deque
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OCTO_PATH = "/media/zhourui/conrft/octo"
CONRFT_EXAMPLES_PATH = os.path.join(REPO_ROOT, "conrft", "examples")

if OCTO_PATH not in sys.path: sys.path.append(OCTO_PATH)
if CONRFT_EXAMPLES_PATH not in sys.path: 
    sys.path.append(CONRFT_EXAMPLES_PATH)
    sys.path.append(os.path.join(REPO_ROOT, "conrft", "scripts"))

from octo.model.octo_model import OctoModel
from data_util import add_mc_returns_to_trajectory, add_next_embeddings_to_trajectory

def load_octo_model(model_path: str, task_desc: str):
    model = OctoModel.load_pretrained(model_path)
    tasks = model.create_tasks(texts=[task_desc])
    return model, tasks

def add_embeddings_to_trajectory(trajectory: List[Dict[str, Any]], model, tasks) -> List[Dict[str, Any]]:
    for i in range(len(trajectory)):
        obs = trajectory[i]['observations']
        # Prepare model inputs: image_primary (256x256), image_wrist (128x128)
        observation = {
            "image_primary": obs["side_policy_256"][np.newaxis, ...],
            "image_wrist": obs["wrist_1"][np.newaxis, ...],
            "timestep_pad_mask": np.array([[True, True]]),
        }
        action_embeddings = model.sample_transformer(observation, tasks)
        trajectory[i]['embeddings'] = action_embeddings[:, -1, :]
    return trajectory

def process_image(image: np.ndarray, camera_name: str, env_config) -> np.ndarray:
    if camera_name in env_config.IMAGE_CROP:
        image = env_config.IMAGE_CROP[camera_name](image)
    target_size = (256, 256) if "256" in camera_name else (128, 128)
    return cv2.resize(image, target_size)[..., ::-1] # BGR to RGB


def build_19d_state(pose_curr, pose_prev, frame_curr, hand, dt=0.05) -> np.ndarray:
    # Check whether frame_curr contains the 'left_pose_xyzw' key
    if "left_pose_xyzw" in frame_curr:
        # If present, use left_pose_xyzw for the first 7 dims and zeros for the remaining 12
        pose = np.array(frame_curr["left_pose_xyzw"])  
        state = np.concatenate([pose, np.zeros(12)])  
    else:
        # Otherwise, fall back to the original pose-based state
        vel = (pose_curr - pose_prev) / dt if pose_prev is not None else np.zeros(6)
        ft = np.zeros(6)  # Force/Torque placeholder
        gripper = np.array([float(frame_curr[f"{hand}_gripper"])])  
        state = np.concatenate([pose_curr, vel, ft, gripper])  
    
    return state


def binarize_gripper_trajectory(gripper_poses):
    """
    Convert a continuous gripper-position sequence to binary 0/1 states.
    Uses backward carry logic: mid-range values depend on the future settled state.
    """
    T = len(gripper_poses)
    new_states = np.zeros(T)
    
    # Initialize carry from the last frame state
    last_pose = gripper_poses[-1]
    g_max = max(gripper_poses)
    g_min = min(gripper_poses)
    eps = 1e-3
    carry = 1.0 if last_pose > g_max - eps else 0.0
    
    # Backward pass
    for i in reversed(range(T)):
        if gripper_poses[i] > g_max - eps:
            carry = 1.0
        elif gripper_poses[i] < g_min + eps:
            carry = 0.0
        else:
            carry = carry
        new_states[i] = carry
    return new_states

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-raw", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--hand", default="left")
    ap.add_argument("--delta-gripper", default=False, action="store_true")
    ap.add_argument("--fix-gripper", default=False, action="store_true",
                help="If set, keep raw gripper values (no binarize/delta)")
    ap.add_argument("--fix-gripper-state", type=float, default=1.0)
    ap.add_argument("--stack-obs-num", type=int, default=2)
    ap.add_argument("--action-def", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--octo-model-path", required=True)
    args = ap.parse_args()
    if args.fix_gripper and args.delta_gripper:
        raise ValueError("--fix-gripper and --delta-gripper cannot be used together.")


    # Load config
    spec = importlib.util.spec_from_file_location("config", args.config)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    train_cfg, env_cfg = cfg_mod.TrainConfig(), cfg_mod.EnvConfig()

    # Load Octo model
    model, tasks = load_octo_model(args.octo_model_path, getattr(train_cfg, 'task_desc', "task"))

    files = sorted(glob.glob(os.path.join(args.input_raw, "**", "*.pkl.gz"), recursive=True))
    all_transitions = []

    for path in tqdm(files, desc="Processing Episodes"):
        with gzip.open(path, 'rb') as f:
            frames = pickle.load(f)["data"]
        T = len(frames)
        if T < 2: continue

        # Unwrapping
        raw_poses = np.array([np.array(f[f"{args.hand}_pose"]).flatten() for f in frames])
        unwrapped = raw_poses.copy()
        for i in range(3, 6): unwrapped[:, i] = np.unwrap(raw_poses[:, i])
        
        grippers = [float(f[f"{args.hand}_gripper"]) for f in frames]

        if args.fix_gripper:
            gripper_series = np.full(len(grippers), float(args.fix_gripper_state), dtype=float)

        else:
            gripper_series = binarize_gripper_trajectory(grippers).astype(float)


        # Build observation sequence (with stacking)
        episode_obs = []
        st_deq = deque(maxlen=args.stack_obs_num)
        img_deqs = {k: deque(maxlen=args.stack_obs_num) for k in ["side_policy_256", "wrist_1", "side_classifier", "demo"]}
        
        for t in range(T):
            s_single = build_19d_state(unwrapped[t], unwrapped[t-1] if t>0 else None, frames[t], args.hand)
            img_dict = {
                "side_policy_256": process_image(frames[t]["front_image"], "side_policy_256", env_cfg),
                "wrist_1": process_image(frames[t][f"{args.hand}_image"], "wrist_1", env_cfg),
                "side_classifier": np.zeros((128, 128, 3), dtype=np.uint8),
                "demo": process_image(frames[t]["right_image"], "demo", env_cfg)
            }
            
            # First-frame padding
            if t == 0:
                for _ in range(args.stack_obs_num):
                    st_deq.append(s_single)
                    for k, v in img_dict.items(): img_deqs[k].append(v)
            else:
                st_deq.append(s_single)
                for k, v in img_dict.items(): img_deqs[k].append(v)
            
            obs = {"state": np.stack(list(st_deq), axis=0)}
            for k in img_dict: obs[k] = np.stack(list(img_deqs[k]), axis=0)
            episode_obs.append(obs)

        traj = []
        for t in range(T):
            done = (t == T - 1)
            # Action processing
            if not done:
                p_act = unwrapped[t+1] if args.action_def == "absolute" else (unwrapped[t+1] - unwrapped[t])
                if args.delta_gripper:
                    g_act = gripper_series[t+1] - gripper_series[t]
                else:
                    g_act = gripper_series[t]
                action = np.concatenate([p_act, [float(g_act)]])

                # g_act = 1.0 if grippers[t] >= g_thresh else 0.0
                action = np.concatenate([p_act, [g_act]])
            else:
                action = traj[-1]['actions'].copy() if traj else np.zeros(7)

            info: Dict[str, Any] = {
                "succeed": done,  
                "intervene_action": action.copy(),  
                "left": 0,  
                "right": 0,  

                "original_state_obs": {
                    "tcp_pose": np.zeros(7, dtype=float), 
                    "tcp_vel": np.zeros(6, dtype=float),  
                    "gripper_pose": np.zeros(1, dtype=float),  
                    "tcp_force": np.zeros(3, dtype=float),     
                    "tcp_torque": np.zeros(3, dtype=float)     
                },
                "grasp_penalty": 0.0  
            }
            if t == 0:
                next_obs = episode_obs[t]
            else:
                next_obs = episode_obs[t]
            traj.append({
                "observations": episode_obs[t],
                "next_observations": episode_obs[t+1] if t < T - 1 else episode_obs[t],
                "actions": action.astype(float),
                "rewards": 10.0 if done else -0.05,
                "masks": 0.0 if done else 1.0,
                "dones": done,
                "infos": info,
            })
        if done:
            if info["succeed"]:
                traj = add_mc_returns_to_trajectory(traj, gamma=train_cfg.discount, reward_scale=1.0, reward_bias=0.0, 
                                                    reward_neg=train_cfg.reward_neg, is_sparse_reward=True)
                traj = add_embeddings_to_trajectory(traj, model, tasks)
                traj = add_next_embeddings_to_trajectory(traj)
        all_transitions.extend(traj)

    with open(args.output, "wb") as f:
        pickle.dump(all_transitions, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Processing finished: total {len(all_transitions)} steps.")

if __name__ == "__main__":
    main()
