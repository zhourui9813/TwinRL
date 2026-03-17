from functools import partial
from typing import Optional

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from octo.utils.typing import Data

from serl_launcher.serl_launcher.common.common import default_init
from serl_launcher.serl_launcher.networks.mlp import MLP
from serl_launcher.serl_launcher.utils.jax_utils import extend_and_repeat, append_zero, append_dims


class ValueCritic(nn.Module):
    encoder: nn.Module
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        outputs = self.network(self.encoder(observations), train=train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)

def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        if jnp.ndim(actions) == 3:
            q_values = jax.vmap(
                lambda a: forward(self, observations, a, **kwargs),
                in_axes=1,
                out_axes=-1,
            )(actions)
        else:
            q_values = forward(self, observations, actions, **kwargs)
        return q_values

    return wrapped

def ensemblize(cls, num_qs, out_axes=0):
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
    )

class Critic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    @multiple_action_q_function
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
            
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        inputs = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network(inputs, train=train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1) 
    
class GraspCritic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    output_dim: Optional[int] = 3
    
    @nn.compact
    def __call__(
        self, 
        observations: jnp.ndarray, 
        train: bool = False
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)
        
        outputs = self.network(obs_enc, train=train)
        if self.init_final is not None:
            value = nn.Dense(
                self.output_dim,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(self.output_dim, kernel_init=default_init())(outputs)
        return value # (batch_size, self.output_dim)

class Policy(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    init_final: Optional[float] = None
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed", or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False, non_squash_distribution: bool = False, repeat: int = -1 
    ) -> distrax.Distribution:
            
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)
            
        if repeat > 0:
            obs_enc = extend_and_repeat(obs_enc, 1, repeat)

        outputs = self.network(obs_enc, train=train)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
                stds = jnp.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
                stds = nn.softplus(stds)
            elif self.std_parameterization == "uniform":
                log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
                stds = jnp.exp(log_stds)
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            assert self.std_parameterization == "fixed"
            stds = jnp.array(self.fixed_std)

        # Clip stds to avoid numerical instability
        # For a normal distribution under MaxEnt, optimal std scales with sqrt(temperature)
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution and not non_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution
    
    def get_features(self, observations):
        return self.encoder(observations, train=False, stop_gradient=True)

class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def stddev(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.stddev())

class ConsistencyPolicy(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    t_network: nn.Module
    action_dim: int
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    steps: int = 40
    clip_denoised: bool = True
    
    def setup(self):
        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho)
        
    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = jnp.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas)
    
    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def base_network(self, x_t: jnp.ndarray, sigmas: jnp.ndarray, obs_enc: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)]
        rescaled_t = 1000 * 0.25 * jnp.log(sigmas + 1e-44)
            
        t = self.t_network(rescaled_t)
        outputs = self.network(jnp.concatenate([c_in*x_t, t, obs_enc], axis=1), train=train)

        denoised = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        denoised = c_out * denoised + c_skip * x_t

        return denoised
    
    def get_features(self, observations):
        return self.encoder(observations, train=False, stop_gradient=True)
    
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, x_t: jnp.ndarray = None, sigmas: jnp.ndarray = None, train: bool = False
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)
        
        if obs_enc.ndim == 1:
            obs_enc = jnp.expand_dims(obs_enc, axis=0)
        
        if x_t is None and sigmas is None:
            batch_size = obs_enc.shape[0]
            x_T = jax.random.normal(self.make_rng('noise'), shape=(batch_size, self.action_dim)) * self.sigma_max
            s_in = jnp.ones((x_T.shape[0],), dtype=x_T.dtype)
            x_0 = self.base_network(x_T, self.sigmas[0] * s_in, obs_enc, train)
        else:
            x_0 = self.base_network(x_t, sigmas, obs_enc, train)
            
        if self.clip_denoised:
            x_0 = jnp.clip(x_0, -1, 1)
            
        return x_0
    
class ConsistencyPolicy_octo(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    t_network: nn.Module
    action_dim: int
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    steps: int = 40
    clip_denoised: bool = True
    
    def setup(self):
        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho)
        
    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = jnp.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas)
    
    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
        c_out = ((sigma - self.sigma_min) * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5)
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def base_network(self, x_t: jnp.ndarray, sigmas: jnp.ndarray, obs_enc: jnp.ndarray, repeat: int = -1, train: bool = False) -> jnp.ndarray:
        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)]
        rescaled_t = 1000 * 0.25 * jnp.log(sigmas + 1e-44)
            
        t = self.t_network(rescaled_t)
        cont_axis = 1
        if repeat > 1:
            t = extend_and_repeat(t, 1, repeat)
            cont_axis = 2
            
        outputs = self.network(jnp.concatenate([c_in*x_t, t, obs_enc], axis=cont_axis), train=train)

        denoised = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        denoised = c_out * denoised + c_skip * x_t

        return denoised
    
    def get_features(self, observations):
        return self.encoder(observations, stop_gradient=True)
    
    @nn.compact
    def __call__(
        self, 
        tasks: Data, 
        observations: jnp.ndarray, 
        action_embeddings: jnp.ndarray = None, 
        x_t: jnp.ndarray = None, 
        sigmas: jnp.ndarray = None, 
        repeat: int = -1,
        train: bool = False,
        stop_octo_gradient: bool = True
    ) -> jnp.ndarray:
        assert self.encoder is not None
        obs_enc, action_embeddings = self.encoder(observations, tasks=tasks, action_embeddings=action_embeddings, train=False, stop_gradient=stop_octo_gradient)
        
        if obs_enc.ndim == 1:
            obs_enc = jnp.expand_dims(obs_enc, axis=0)
            
        if repeat > 1:
            obs_enc = extend_and_repeat(obs_enc, 1, repeat)
        
        if x_t is None and sigmas is None:
            batch_size = obs_enc.shape[0]
            x_shape = (batch_size, repeat, self.action_dim) if repeat > 1 else (batch_size, self.action_dim)
            x_T = jax.random.normal(self.make_rng('noise'), shape=x_shape) * self.sigma_max
            s_in = jnp.ones((batch_size,), dtype=x_T.dtype)
            x_0 = self.base_network(x_T, self.sigmas[0] * s_in, obs_enc, repeat, train)
        else:
            x_0 = self.base_network(x_t, sigmas, obs_enc, repeat, train)
                
        if self.clip_denoised:
            x_0 = jnp.clip(x_0, -1, 1)
            
        return x_0, action_embeddings
    