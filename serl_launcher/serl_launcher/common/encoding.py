from typing import Dict, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from octo.model.octo_module import OctoTransformer
from octo.utils.typing import Config, Data, Params, PRNGKey, Sequence

class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        T = image.shape[0]
                        if T > 1:
                            image = image[-1:] # for stacked images, only use the last frame
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        T = image.shape[1]
                        if T > 1:
                            image = image[:, -1:] # for stacked images, only use the last frame
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded
    
class OctoEncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: OctoTransformer
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        tasks: Data=None,
        action_embeddings: jnp.ndarray=None,
        train=True,
        stop_gradient=False,
    ) -> jnp.ndarray:   
        if action_embeddings is None:     
            image_primary = observations["side_policy_256"]
            image_wrist = observations["wrist_1"]
            if image_primary.ndim == 4:
                image_primary = image_primary[jnp.newaxis, ...]
                image_wrist = image_wrist[jnp.newaxis, ...]
            batch_size = image_primary.shape[0]
            window_size = image_primary.shape[1]
            timestep_pad_mask = jnp.ones((batch_size, window_size), dtype=bool)
            
            if not stop_gradient:
                def mask_image(image, mask_flag):
                    return jax.lax.cond(
                        mask_flag,
                        lambda _: jnp.zeros_like(image),
                        lambda _: image, 
                        operand=None)
                
                mask_flags = jax.random.bernoulli(self.make_rng('mask_wrist'), p=0.0, shape=(batch_size,))
                image_wrist = jax.vmap(mask_image)(image_wrist, mask_flags)
            
            observation_octo = {"image_primary": image_primary,
                "image_wrist": image_wrist, 
                "timestep_pad_mask": timestep_pad_mask,
            }

            transformer_outputs = self.encoder(observation_octo, tasks, timestep_pad_mask, train=not stop_gradient)
            token_group = transformer_outputs["readout_action"]
            action_embeddings = token_group.tokens.mean(axis=-2)
            
            action_embeddings = action_embeddings[:, -1, :] # remove window_size dimension
        else:
            action_embeddings = action_embeddings
            
        if stop_gradient:
            action_embeddings = jax.lax.stop_gradient(action_embeddings)

        encoded = action_embeddings
        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded, action_embeddings


