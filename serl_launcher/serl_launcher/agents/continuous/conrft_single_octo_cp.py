from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import numpy as np
import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.serl_launcher.common.encoding import EncodingWrapper, OctoEncodingWrapper
from serl_launcher.serl_launcher.common.optimizers import make_optimizer
from serl_launcher.serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.serl_launcher.networks.actor_critic_nets import Critic, Policy, ConsistencyPolicy_octo, ensemblize
from serl_launcher.serl_launcher.networks.mlp import MLP, timeMLP
from serl_launcher.serl_launcher.utils.train_utils import _unpack
from serl_launcher.serl_launcher.utils.train_utils import _unpack, get_weightings, get_snr
from serl_launcher.serl_launcher.utils.jax_utils import append_dims, mean_flat

from octo.model.octo_model import OctoModel


class ConrftCPOctoAgentSingleArm(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        action_embeddings: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        action_embeddings: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations,
            action_embeddings,
            actions,
            rng=rng,
            grad_params=self.state.target_params
        )

    def forward_policy(
        self,
        tasks: Data,
        observations: Data,
        action_embeddings: Data = None,
        x_t: Data = None,
        sigmas: Data = None,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
        repeat: int = -1,
        stop_octo_gradient: bool = False,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        # rng, noise_rng = jax.random.split(rng, 2)
        rng, noise_rng, mask_rng = jax.random.split(rng, 3)
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            tasks,
            observations,
            action_embeddings,
            x_t,
            sigmas,
            repeat,
            name="actor",
            rngs={"dropout": rng, "noise": noise_rng, "mask_wrist": mask_rng} if train else {
                "noise": noise_rng},
            train=train,
            stop_octo_gradient=stop_octo_gradient,
        )

    def forward_policy_and_sample(
        self,
        tasks: Data,
        obs: Data,
        action_embeddings: Data = None,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        repeat=None,
        **kwargs,
    ):
        rng, sample_rng = jax.random.split(rng)
        new_actions, _ = self.forward_policy(
            tasks, obs, action_embeddings, repeat=repeat, rng=rng, grad_params=grad_params, train=True)

        return new_actions

    def _compute_next_actions(self, batch, rng, repeat=-1):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_actions, _ = self.forward_policy(
            batch["tasks"], batch["next_observations"], None, rng=rng, repeat=repeat,)

        return next_actions

    def _get_cql_q_diff(self, batch, rng: PRNGKey, grad_params: Optional[Params] = None):
        """
        most of the CQL loss logic is here
        It is needed for both critic_loss_fn and cql_alpha_loss_fn
        """
        info = {}
        batch_size = batch["rewards"].shape[0]
        actions = batch["actions"][..., :-
                                   1] if self.config["fix_gripper"] else batch["actions"]

        rng, critic_rng = jax.random.split(rng)
        q_pred = self.forward_critic(
            batch['observations'], batch["embeddings"], actions, critic_rng, grad_params=grad_params,)
        chex.assert_shape(
            q_pred, (self.config["critic_ensemble_size"], batch_size))

        """sample random actions"""
        rng, action_rng = jax.random.split(rng)
        if self.config["cql_action_sample_method"] == "uniform":
            cql_random_actions = jax.random.uniform(action_rng, shape=(
                batch_size, self.config["cql_n_actions"], self.config["action_dim"]), minval=-1.0, maxval=1.0,)
        elif self.config["cql_action_sample_method"] == "normal":
            cql_random_actions = jax.random.normal(action_rng, shape=(
                batch_size, self.config["cql_n_actions"], self.config["action_dim"]),)
        else:
            raise NotImplementedError

        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        cql_current_actions = self.forward_policy_and_sample(
            batch["tasks"], batch['observations'], None, current_a_rng, repeat=self.config["cql_n_actions"],)
        chex.assert_shape(cql_current_actions, (batch_size,
                          self.config["cql_n_actions"], self.config["action_dim"]),)

        cql_next_actions = self.forward_policy_and_sample(
            batch["tasks"], batch['next_observations'], None, next_a_rng, repeat=self.config["cql_n_actions"],)

        # all_sampled_actions follows the order of [random, current, next]
        all_sampled_actions = jnp.concatenate(
            [cql_random_actions, cql_current_actions, cql_next_actions,], axis=1,)

        """q values of randomly sampled actions"""
        rng, q_rng = jax.random.split(rng)
        cql_q_samples = self.forward_critic(
            batch["observations"], batch["embeddings"], all_sampled_actions, q_rng, grad_params=grad_params)
        chex.assert_shape(
            cql_q_samples, (self.config["critic_ensemble_size"], batch_size, self.config["cql_n_actions"] * 3,),)

        info["all_sampled_action_values"] = cql_q_samples.mean()
        info["random_action_values"] = cql_q_samples[:,
                                                     :, : self.config["cql_n_actions"]].mean()
        info["current_action_values"] = cql_q_samples[:, :,
                                                      self.config["cql_n_actions"]: 2 * self.config["cql_n_actions"]].mean()
        info["next_action_values"] = cql_q_samples[:,
                                                   :, 2 * self.config["cql_n_actions"]:].mean()

        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            cql_q_samples = cql_q_samples[subsample_idcs]
            q_pred = q_pred[subsample_idcs]
            critic_size = self.config["critic_subsample_size"]
        else:
            critic_size = self.config["critic_ensemble_size"]

        """Cal-QL"""
        n_actions_for_calql = self.config["cql_n_actions"] * 3
        mc_lower_bound = jnp.repeat(
            batch['mc_returns'].reshape(-1, 1), n_actions_for_calql, axis=1)
        chex.assert_shape(mc_lower_bound, (batch_size, n_actions_for_calql))

        num_vals = jnp.size(cql_q_samples[:, :, :n_actions_for_calql])
        calql_bound_rate = jnp.sum(cql_q_samples < mc_lower_bound) / num_vals
        cql_q_samples = jnp.maximum(cql_q_samples, mc_lower_bound)

        # cql_importance_sample
        assert self.config["cql_importance_sample"] is False

        cql_q_samples = jnp.concatenate(
            [cql_q_samples, jnp.expand_dims(q_pred, -1),], axis=-1,)
        cql_q_samples -= jnp.log(cql_q_samples.shape[-1]
                                 ) * self.config["cql_temp"]
        chex.assert_shape(cql_q_samples, (critic_size, batch_size,
                          self.config["cql_n_actions"] * 3 + 1,),)

        """log sum exp of the ood actions"""
        cql_ood_values = (jax.scipy.special.logsumexp(
            cql_q_samples / self.config["cql_temp"], axis=-1) * self.config["cql_temp"])
        chex.assert_shape(cql_ood_values, (critic_size, batch_size))

        cql_q_diff = cql_ood_values - q_pred
        info["cql_ood_values"] = cql_ood_values.mean()
        info["calql_bound_rate"] = calql_bound_rate

        return cql_q_diff, info

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        actions = batch["actions"][..., :-
                                   1] if self.config["fix_gripper"] else batch["actions"]

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions = self._compute_next_actions(
            batch, next_action_sample_key)

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"], batch["next_embeddings"], next_actions, rng=rng,)  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (batch["rewards"] + self.config["discount"]
                    * batch["masks"] * target_next_min_q)
        chex.assert_shape(target_q, (batch_size,))

        predicted_qs = self.forward_critic(
            batch["observations"], batch["embeddings"], actions, rng=rng, grad_params=params)

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size))
        target_qs = target_q[None].repeat(
            self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info

    def calql_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        td_loss, td_loss_info = self.critic_loss_fn(batch, params, rng)

        cql_q_diff, cql_intermediate_results = self._get_cql_q_diff(
            batch, rng, params)

        alpha = self.config["cql_alpha"]
        cql_loss = jnp.clip(
            cql_q_diff, self.config["cql_clip_diff_min"], self.config["cql_clip_diff_max"],).mean()

        critic_loss = td_loss + alpha * cql_loss
        info = {
            **td_loss_info,
            "critic_loss": critic_loss,
            "td_loss": td_loss,
            "cql_loss": cql_loss,
            "cql_alpha": alpha,
            "cql_diff": cql_q_diff.mean(),
            **cql_intermediate_results,
        }

        return critic_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        # Consistency loss
        rng, noise_rng, indice_rng, policy_rng1, policy_rng2, policy_rng3, critic_rng = jax.random.split(
            rng, 7)

        new_actions, action_embeddings = self.forward_policy(
            batch["tasks"], batch["observations"], None, rng=policy_rng1, grad_params=params)

        actions = batch["actions"][..., :-
                                   1] if self.config["fix_gripper"] else batch["actions"]
        x_start = actions
        noise = jax.random.normal(
            noise_rng, shape=x_start.shape, dtype=x_start.dtype)
        dims = x_start.ndim

        indices = jax.random.randint(
            indice_rng, (batch_size,), 0, self.config["num_scales"]-1)

        t = self.config["sigma_max"] ** (1 / self.config["rho"]) + indices / (self.config["num_scales"] - 1) * (
            self.config["sigma_min"] ** (1 / self.config["rho"]) -
            self.config["sigma_max"] ** (1 / self.config["rho"])
        )
        t = t**self.config["rho"]

        x_t = x_start + noise * append_dims(t, dims)

        distiller, _ = self.forward_policy(
            batch["tasks"], batch["observations"], None, x_t, t, rng=policy_rng2, grad_params=params)

        snrs = get_snr(t)
        weights = get_weightings("karras", snrs, self.config["sigma_data"])

        recon_diffs = (distiller - x_start) ** 2
        recon_loss = (mean_flat(recon_diffs) * weights).mean()

        mse = ((new_actions - actions) ** 2).sum(-1)
        q_new_actions = self.forward_critic(
            batch["observations"], batch["embeddings"], new_actions, rng=critic_rng,)
        q_new_actions = q_new_actions.mean(axis=0)
        chex.assert_shape(q_new_actions, (batch_size,))

        q_loss = - q_new_actions.mean()

        actor_loss = self.state.bc_weight * recon_loss + self.state.q_weight * q_loss

        info = {
            "actor_loss": actor_loss,
            "q_weight": self.state.q_weight,
            "bc_weight": self.state.bc_weight,
            "q_loss": q_new_actions.mean(),
            "bc_loss": recon_loss,
            "mse": mse.mean(),
        }

        return actor_loss, info

    def calql_loss_fns(self, batch):
        losses = {
            "actor": partial(self.policy_loss_fn, batch),
            "critic": partial(self.calql_critic_loss_fn, batch),
        }

        return losses

    def loss_fns(self, batch):
        losses = {
            "actor": partial(self.policy_loss_fn, batch),
            "critic": partial(self.critic_loss_fn, batch),
        }

        return losses

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update_calql(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic"}),
        **kwargs
    ) -> Tuple["ConrftCPOctoAgentSingleArm", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks", "mc_returns".
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """

        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 7))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]})

        # Compute gradients and update params
        calql_loss_fns = self.calql_loss_fns(batch, **kwargs)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            calql_loss_fns.keys()), f"Invalid gradient steps: {networks_to_update}"
        for key in calql_loss_fns.keys() - networks_to_update:
            calql_loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            calql_loss_fns, pmap_axis=pmap_axis, has_aux=True)

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(
                self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (hasattr(opt_state, "hyperparams") and "learning_rate" in opt_state.hyperparams.keys()):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update_ql(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic"}),
        **kwargs
    ) -> Tuple["ConrftCPOctoAgentSingleArm", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks", "mc_returns".
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """

        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 7))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]})

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, **kwargs)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True)

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(
                self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (hasattr(opt_state, "hyperparams") and "learning_rate" in opt_state.hyperparams.keys()):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit)
    def sample_actions(
        self,
        observations: Data,
        tasks: Data,
        *,
        seed: Optional[PRNGKey] = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """

        actions, action_embeddings = self.forward_policy(
            tasks, observations, rng=seed, train=False)
        actions = jnp.squeeze(actions, axis=0)

        if self.config["fix_gripper"]:  # add gripper action, default to 0
            actions = jnp.concatenate([actions, jnp.array([0])])

        return actions, action_embeddings

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        tasks: Data,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        fix_gripper: bool = False,
        # Algorithm config
        num_scales: int = 40,
        sigma_min: float = 0.02,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        cql_n_actions: int = 10,
        entropy_per_dim: bool = False,
        cql_temp: float = 1.0,
        cql_action_sample_method: str = "uniform",
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        cql_alpha: float = 0.1,
        cql_importance_sample: bool = False,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        q_weight: float = 0,
        bc_weight: float = 1.0,
        bc_weight_rate: float = 5e-5,
        bc_weight_min: float = 0.05,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
        }

        rng, init_rng, noise_rng = jax.random.split(rng, 3)
        init_rng = {"params": init_rng, "noise": noise_rng}

        params = model_def.init(
            init_rng,
            actor=[tasks, observations],
            critic=[observations, actions[:-1] if fix_gripper else actions],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
            bc_weight=bc_weight,
            q_weight=q_weight,
        )

        # Config
        action_dim = actions.shape[-1] - \
            1 if fix_gripper else actions.shape[-1]
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = - action_dim / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                fix_gripper=fix_gripper,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                cql_action_sample_method=cql_action_sample_method,
                cql_n_actions=cql_n_actions,
                action_dim=action_dim,
                cql_temp=cql_temp,
                cql_clip_diff_min=cql_clip_diff_min,
                cql_clip_diff_max=cql_clip_diff_max,
                cql_alpha=cql_alpha,
                cql_importance_sample=cql_importance_sample,
                bc_weight_min=bc_weight_min,
                bc_weight_rate=bc_weight_rate,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                num_scales=num_scales,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_data=sigma_data,
                rho=rho,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        tasks: Data,
        octo_model: OctoModel,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        fix_gripper: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_t_network_kwargs: dict = {
            "t_dims": 16,
        },
        policy_kwargs: dict = {
            "clip_denoised": True,
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        q_weight: float = 0.1,
        bc_weight: float = 1.0,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "resnet":
            from serl_launcher.serl_launcher.vision.resnet_v1 import resnetv1_configs
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
            from serl_launcher.serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder, resnetv1_configs)

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

        critic_encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        actor_encoder_def = OctoEncodingWrapper(
            encoder=octo_model.module.octo_transformer,
            use_proprio=use_proprio,
            enable_stacking=True,
        )

        encoders = {
            "critic": critic_encoder_def,
            "actor": actor_encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(
            critic_backbone, critic_ensemble_size)(name="critic_ensemble")
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone)(name="critic")

        actor_def = ConsistencyPolicy_octo(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            t_network=timeMLP(**policy_t_network_kwargs),
            action_dim=actions.shape[-1] -
            1 if fix_gripper else actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=actor_def,
            critic_def=critic_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            tasks=tasks,
            fix_gripper=fix_gripper,
            q_weight=q_weight,
            bc_weight=bc_weight,
            **kwargs,
        )

        # load pretrained weights for ResNet-10
        if "pretrained" in encoder_type:
            from serl_launcher.serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        # load pretrained weights for Octo
        new_params = agent.state.params
        new_params["modules_actor"]["encoder"]["encoder"] = octo_model.params["octo_transformer"]
        agent = agent.replace(state=agent.state.replace(params=new_params))

        return agent
