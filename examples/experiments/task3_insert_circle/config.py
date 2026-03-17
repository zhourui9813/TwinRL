import os
import jax
import numpy as np
import jax.numpy as jnp
import threading

from serl_robot_infra.franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from serl_robot_infra.franka_env.envs.relative_env import RelativeFrame
from serl_robot_infra.franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.serl_launcher.networks.reward_classifier import load_classifier_func

from examples.experiments.config import DefaultTrainingConfig
from examples.experiments.task1_pick_banana_basket.wrapper import PickBananaEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.1:5000/"
    DEVICE = "SpaceMouse Wireless"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "242322070237",
            "dim": (640, 480),
            "exposure": 10500,
        },
        "side_policy_256": {
            "serial_number": "341222301576",
            "dim": (640, 480),
            "exposure": 13000,
        },
        "side_classifier": {
            "serial_number": "341222301576", 
            "dim": (640, 480),
            "exposure": 13000,
        },
        "demo": {
            "serial_number": "242322070237", 
            "dim": (640, 480),
            "exposure": 13000,
        },
    }
    IMAGE_CROP = {"wrist_1": lambda img: img,
                  "side_policy_256": lambda img: img,
                  "side_classifier": lambda img: img,
                  "demo": lambda img: img}

    # TARGET_POSE = np.array([0.33, -0.15, 0.20, np.pi, 0, 0])
    TARGET_POSE = np.array([0.5196274163493255,-0.2569823226322744,0.15784099682936825,2.951380580275543,-0.024769121183364717,-0.02074006537640294])
    # RESET_POSE = np.array([0.61, -0.17, 0.22, np.pi, 0, 0])
    RESET_POSE = np.array([0.48049184958385605,-0.00022334620358853869,0.30009448728042004,3.140056888974983,0.007291706929760888,0.006764377533135413])
    ACTION_SCALE = np.array([0.08, 0.2, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    # RANDOM_XY_RANGE = 0.02
    # RANDOM_RZ_RANGE = 0.03

    RANDOM_XY_RANGE = 0.00
    RANDOM_RZ_RANGE = 0.00
    # ABS_POSE_LIMIT_HIGH = RESET_POSE + np.array([0.2, 0.2, 0.1, 0.05, 0.1, 0.1])
    # ABS_POSE_LIMIT_LOW = RESET_POSE - np.array([0.05, 0.2, 0.5, 0.1, 0.1, 0.1])

    ABS_POSE_LIMIT_HIGH = RESET_POSE + np.array([0.5, 0.5, 0.3, 0.05, 0.1, 0.1])
    ABS_POSE_LIMIT_LOW = RESET_POSE - np.array([0.2, 0.5, 0.5, 0.1, 0.1, 0.1])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.005, 
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005, 
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02, 
        "rotational_Ki": 0,
    }  # for normal operation other than reset procedure
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }  # only for reset procedure
    MAX_EPISODE_LENGTH = 100
 

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy_256", "wrist_1"]
    classifier_keys = ["side_classifier"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "Insert the circle."
    octo_path = "/media/zhourui/Twin-RL/model_hub/octo-small"

    def __init__(self):
        super().__init__()
        self._reward_flag = threading.Event()

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        env = PickBananaEnv( fake_env=fake_env, save_video=save_video, config=EnvConfig())
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        if classifier:
            def reward_func(obs):
                def sigmoid(x): return 1 / (1 + jnp.exp(-x))
                # Should open the gripper and pull up after putting the banana
                print(f"classifier(obs): {sigmoid(classifier(obs)[0])}")
                print(f"env.curr_gripper_pos: {env.curr_gripper_pos}")
                print(f"env.currpos: {env.currpos}")

                if sigmoid(classifier(obs)[0]) > 0.95 and env.curr_gripper_pos > 0.045 and env.currpos[2] > 0.035:
                    return 10.0
                else:
                    return self.reward_neg

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.2)
        return env
