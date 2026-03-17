from functools import partial
from typing import Any, Iterable, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.networks.actor_critic_nets import Policy
from serl_launcher.networks.diffusion_nets import (
    FourierFeatures,
    ScoreActor,
    cosine_beta_schedule,
    vp_beta_schedule,
)
from serl_launcher.networks.mlp import MLP, MLPResNet
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop

def ddpm_bc_loss(noise_prediction, noise):
    ddpm_loss = jnp.square(noise_prediction - noise).sum(-1)

    return ddpm_loss.mean(), {
        "ddpm_loss": ddpm_loss,
        "ddpm_loss_mean": ddpm_loss.mean(),
    }


class DDPMBCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def data_augmentation_fn(self, rng, observations):
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        def loss_fn(params, rng):
            key, rng = jax.random.split(rng)
            time = jax.random.randint(
                key, (batch["actions"].shape[0],), 0, self.config["diffusion_steps"]
            )
            key, rng = jax.random.split(rng)
            noise_sample = jax.random.normal(key, batch["actions"].shape)

            alpha_hats = self.config["alpha_hats"][time]
            time = time[:, None]
            alpha_1 = jnp.sqrt(alpha_hats)[:, None, None]
            alpha_2 = jnp.sqrt(1 - alpha_hats)[:, None, None]

            noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

            rng, key = jax.random.split(rng)
            noise_pred = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                batch["observations"],
                noisy_actions,
                time,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )

            return ddpm_bc_loss(
                noise_pred,
                noise_sample,
            )

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        clip_sampler: bool = True,
        repeat: int = 1,
    ) -> jnp.ndarray:        
        def fn(input_tuple, time):
            current_x, rng = input_tuple
            input_time = jnp.broadcast_to(time, (current_x.shape[0], 1))

            eps_pred = self.state.apply_fn(
                {"params": self.state.target_params},
                observations,
                current_x,
                input_time,
                name="actor",
            )

            alpha_1 = 1 / jnp.sqrt(jnp.array(self.config["alphas"])[time])
            alpha_2 = (1 - jnp.array(self.config["alphas"])[time]) / (
                jnp.sqrt(1 - jnp.array(self.config["alpha_hats"])[time])
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(
                key,
                shape=current_x.shape,
            )
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (
                jnp.sqrt(jnp.array(self.config["betas"])[time]) * z_scaled
            )

            if clip_sampler:
                current_x = jnp.clip(
                    current_x, self.config["action_min"], self.config["action_max"]
                )

            return (current_x, rng), ()
        
        key, rng = jax.random.split(seed)
        
        observations = jax.tree.map(
            lambda x: jnp.repeat(x, repeat, axis=1).reshape(
                repeat, 1, *x.shape[2:]
            ),
            observations,
        )

        input_tuple, () = jax.lax.scan(
            fn,
            (
                jax.random.normal(
                    key, (repeat, *self.config["action_dim"])
                ),
                rng,
            ),
            jnp.arange(self.config["diffusion_steps"] - 1, -1, -1),
        )
        
        for _ in range(self.config["repeat_last_step"]):
            input_tuple, () = fn(input_tuple, 0)
            
        action_0, rng = input_tuple
        action_0 = action_0.reshape(1, repeat, -1)
        
        return action_0
        
    @jax.jit
    def get_debug_metrics(self, batch, seed, gripper_close_val=None):
        actions = self.sample_actions(observations=batch["observations"], seed=seed)

        metrics = {
            "mse": ((actions - batch["actions"]) ** 2).sum((-2, -1)).mean(),
        }

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        score_network_kwargs: dict = {
            "time_dim": 32,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "hidden_dim": 256,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        augmentation_function: Optional[callable] = None,
        # DDPM algorithm train + inference config
        beta_schedule: str = "cosine",
        diffusion_steps: int = 25,
        action_samples: int = 1,
        repeat_last_step: int = 0,
    ):
        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        network_kwargs["activate_final"] = True
        networks = {
            "actor": ScoreActor(
                encoder_def,
                FourierFeatures(score_network_kwargs["time_dim"], learnable=True),
                MLP(
                    (
                        2 * score_network_kwargs["time_dim"],
                        score_network_kwargs["time_dim"],
                    )
                ),
                MLPResNet(
                    score_network_kwargs["num_blocks"],
                    actions.shape[-1],
                    dropout_rate=score_network_kwargs["dropout_rate"],
                    use_layer_norm=score_network_kwargs["use_layer_norm"],
                ),
            ),
        }

        model_def = ModuleDict(networks)

        tx = optax.adam(learning_rate)
        
        example_time = jnp.zeros((1,))

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng, 
            actor=[observations, actions, example_time]
            )["params"]
        
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )
        
        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(diffusion_steps))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, diffusion_steps)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(diffusion_steps))
        
        alphas = 1 - betas
        alpha_hat = jnp.array(
            [jnp.prod(alphas[: i + 1]) for i in range(diffusion_steps)]
        )
        
        config = dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            diffusion_steps=diffusion_steps,
            action_samples=action_samples,
            repeat_last_step=repeat_last_step,
        )

        agent = cls(state, config)

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
