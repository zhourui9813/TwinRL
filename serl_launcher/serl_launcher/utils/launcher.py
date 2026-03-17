# !/usr/bin/env python3

import jax
from jax import nn
import jax.numpy as jnp

from agentlace.trainer import TrainerConfig

from serl_launcher.serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.serl_launcher.common.wandb import WandBLogger
from serl_launcher.serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.serl_launcher.agents.continuous.sac_single import SACAgentSingleArm
from serl_launcher.serl_launcher.agents.continuous.conrft_single_octo_cp import ConrftCPOctoAgentSingleArm
from serl_launcher.serl_launcher.vision.data_augmentations import batched_random_crop

##############################################################################


def make_bc_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    fix_gripper: bool = False,
):
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        fix_gripper=fix_gripper,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_sac_pixel_agent_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_conrft_octo_cp_pixel_agent_single_arm(
    seed,
    sample_obs,
    sample_action,
    sample_tasks,
    octo_model,
    encoder_type="resnet-pretrained",
    image_keys=("image",),
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    num_scales=40,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    fix_gripper: bool = False,
    q_weight: float = 0.1,
    bc_weight: float = 1.0,
    optimizer_warmup_steps: int = 2000,
    optimizer_cosine_decay_steps: int = 50_000,
    optimizer_peak_learning_rate: float = 3e-5,
):
    print(f"optimizer_warmup_steps: {optimizer_warmup_steps}")
    print(f"optimizer_cosine_decay_steps: {optimizer_cosine_decay_steps}")
    print(f"optimizer_peak_learning_rate: {optimizer_peak_learning_rate}")
    agent = ConrftCPOctoAgentSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        sample_tasks,
        encoder_type=encoder_type,
        use_proprio=True,
        octo_model=octo_model,
        image_keys=image_keys,
        fix_gripper=fix_gripper,
        policy_kwargs={
            "sigma_data": sigma_data,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "rho": rho,
            "steps": num_scales,
            "clip_denoised": True,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_t_network_kwargs={
            "t_dim": 16,
            "activations": nn.tanh,
        },
        actor_optimizer_kwargs={
            "learning_rate": optimizer_peak_learning_rate,
            "warmup_steps": optimizer_warmup_steps,          
            "cosine_decay_steps": optimizer_cosine_decay_steps,   
            "weight_decay": 0.001,           
            "clip_grad_norm": 1.0,
        },
        critic_optimizer_kwargs={
            "learning_rate": optimizer_peak_learning_rate,
            "warmup_steps": optimizer_warmup_steps,
            "cosine_decay_steps": optimizer_cosine_decay_steps,
            "weight_decay": 0.001,
            "clip_grad_norm": 1.0,
        },
        num_scales=num_scales,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        rho=rho,
        discount=discount,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        augmentation_function=make_batch_augmentation_func(image_keys),
        q_weight=q_weight,
        bc_weight=bc_weight,
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000

    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + \
        (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value


def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(
            next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch

    return augment_batch


def make_trainer_config(port_number: int = 3333, broadcast_port: int = 3334):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "conrft",
    description: str = "serl_launcher",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
