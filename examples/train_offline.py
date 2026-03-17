#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from flax.core import frozen_dict
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
import sys
from pathlib import Path

from serl_launcher.serl_launcher.agents.continuous.conrft_single_octo_cp import ConrftCPOctoAgentSingleArm
from serl_launcher.serl_launcher.utils.timer_utils import Timer
from serl_launcher.serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.serl_launcher.utils.launcher import (
    make_conrft_octo_cp_pixel_agent_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from examples.experiments.mappings import CONFIG_MAPPING

from octo.model.octo_model import OctoModel

from utils.data_utils import add_mc_returns_to_trajectory, add_next_embeddings_to_trajectory

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0,
                     "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")

flags.DEFINE_float("gamma", 0.95, "return discount")
flags.DEFINE_float("reward_neg", 0.0, "reward_neg for spase reward envs")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_float("q_weight", 0.1, "q_weight ")
flags.DEFINE_float("bc_weight", 1.0, "bc_weight")

flags.DEFINE_integer("pretrain_steps", 2000, "Number of pretrain steps.")

flags.DEFINE_integer("port_number", 3333,
                     "actor and learner port number")
flags.DEFINE_integer("broadcast_port", 3334,
                     "actor and learner broadcast number")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################

def learner(rng, tasks, agent, replay_buffer, demo_buffer, wandb_logger=None, port_number=3333, broadcast_port=3334):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(
            FLAGS.checkpoint_path))[11:]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    step = start_step

    train_critic_networks_to_update = frozenset({"critic"})
    train_actor_networks_to_update = frozenset({"actor"})
    train_networks_to_update = frozenset({"critic", "actor"})

    def create_batch_tasks(data_dict, batch_size):
        batch_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):  # Handling nested dictionary (e.g., language_instruction)
                batch_dict[key] = {k: np.tile(
                    v, (batch_size, *([1] * (v.ndim - 1)))) for k, v in value.items()}
            else:
                # For non-dictionary values, repeat along batch dimension (axis=0)
                batch_dict[key] = np.tile(
                    value, (batch_size, *([1] * (value.ndim - 1))))  # Repeat along axis 0

        return batch_dict

    # Pretrain the model with the demo data
    if step < FLAGS.pretrain_steps:
        print_green("Pretraining the model with demo data")
        for step in tqdm.tqdm(range(start_step, FLAGS.pretrain_steps + 1), desc="pretraining"):
            for _ in range(config.cta_ratio - 1):
                batch = next(demo_buffer.get_iterator(
                    sample_args={"batch_size": config.batch_size,
                                 "pack_obs": True, },
                    device=sharding.replicate(),
                ))

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
                batch = frozen_dict.freeze(batch)
                agent, critics_info = agent.update_calql(
                    batch, networks_to_update=train_critic_networks_to_update,)

            batch = next(demo_buffer.get_iterator(
                sample_args={"batch_size": config.batch_size,
                             "pack_obs": True, },
                device=sharding.replicate(),
            ))

            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)

            agent, update_info = agent.update_calql(
                batch, networks_to_update=train_networks_to_update,)
            
            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=step)

            if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
                ui = jax.device_get(update_info)
                ai = ui["actor"]
                to_float = lambda x: float(np.asarray(x).mean())
                print(f"step={step} actor_loss={to_float(ai['actor_loss'])} q_loss={to_float(ai['q_loss'])} bc_loss={to_float(ai['bc_loss'])} mse={to_float(ai['mse'])}")
                
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path, agent.state, step=step, keep=100)

        print_green("Pretraining done")
        return  # after pretraining, return and exit
    else:
        print_green(
            "Existing pretrained checkpoint model found. Skipping pretraining")
##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner, save_video=FLAGS.eval_checkpoint_step, classifier=True, stack_obs_num=2)
    env = RecordEpisodeStatistics(env)

    FLAGS.reward_neg = config.reward_neg

    rng, sampling_rng = jax.random.split(rng)

    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])

    total_opt_steps = FLAGS.pretrain_steps * config.cta_ratio
    cosine_decay_steps = max(total_opt_steps - config.warmup_steps, 1)

    if config.setup_mode == 'single-arm-fixed-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            fix_gripper=True,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
            optimizer_warmup_steps=config.warmup_steps,
            optimizer_cosine_decay_steps=cosine_decay_steps,
            optimizer_peak_learning_rate=config.pretrain_peak_learning_rate,
        )
        include_grasp_penalty = False
        include_octo_embeddings = True
        include_mc_returns = True
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
            optimizer_warmup_steps=config.warmup_steps,
            optimizer_cosine_decay_steps=cosine_decay_steps,
            optimizer_peak_learning_rate=config.pretrain_peak_learning_rate,
        )
        include_grasp_penalty = True
        include_octo_embeddings = True
        include_mc_returns = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(
        jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        if not FLAGS.learner:
            input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path, agent.state,)
        # agent = agent.replace(state=ckpt)

        # Update params only, ignore the optimizer states
        new_params = ckpt.params
        new_target_params = ckpt.target_params

        agent = agent.replace(state=agent.state.replace(
            params=new_params, target_params=new_target_params))

        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(FLAGS.checkpoint_path))[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        # set up wandb and logging

        wandb_logger = make_wandb_logger(
            project="conrft",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(
            sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        assert FLAGS.demo_path is not None

        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(sampling_rng,
                tasks,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                wandb_logger=wandb_logger,
                port_number=FLAGS.port_number,
                broadcast_port=FLAGS.broadcast_port
                )


if __name__ == "__main__":
    app.run(main)
