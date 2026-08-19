from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _rtc_prefix_weights(
    n_action_steps: int,
    prefix_len: int,
    decay_steps: int,
    schedule: str = "exp",
    *,
    target_len: int | None = None,
) -> np.ndarray:
    """Per-timestep RTC prefix weights (PI-Kinetix get_prefix_weights).

    weights[:prefix_len] = 1 (locked prefix); a decay band of `decay_steps`
    follows, easing 1 -> 0 (exp/linear); the rest is 0 (free future).
    If ``target_len`` is provided, all non-zero weights are clipped to the
    number of real actions supplied by the client. Padding is never a target.
    """
    w = np.zeros(n_action_steps, dtype=np.float32)
    target_end = n_action_steps if target_len is None else min(max(0, target_len), n_action_steps)
    locked = min(max(0, prefix_len), target_end)
    if locked <= 0:
        return w
    w[:locked] = 1.0
    decay_end = min(locked + max(0, decay_steps), target_end)
    decay_len = decay_end - locked
    for i in range(decay_len):
        pos = locked + i
        frac = i / max(decay_len - 1, 1)
        if schedule == "exp":
            w[pos] = float(np.exp(-3.0 * frac))
        elif schedule == "linear":
            w[pos] = 1.0 - frac
        elif schedule == "ones":
            w[pos] = 1.0
        else:  # "zeros" or unknown -> no decay band
            w[pos] = 0.0
    return w


def _prepare_rtc_prefix(
    prefix_norm: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
    lock_len: int,
    decay_steps: int,
    schedule: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pad RTC targets without letting padding become a guidance target."""
    prefix_norm = np.asarray(prefix_norm, dtype=np.float32)
    if prefix_norm.ndim != 2:
        raise ValueError(f"RTC prefix must have shape (L, D), got {prefix_norm.shape}")

    target_len = min(prefix_norm.shape[0], action_horizon)
    lock_len = min(max(0, int(lock_len)), target_len)
    copy_dim = min(prefix_norm.shape[1], action_dim)
    prefix_full = np.zeros((action_horizon, action_dim), dtype=np.float32)
    prefix_full[:target_len, :copy_dim] = prefix_norm[:target_len, :copy_dim]
    weights = _rtc_prefix_weights(
        action_horizon,
        lock_len,
        decay_steps,
        schedule,
        target_len=target_len,
    )
    return prefix_full, weights, lock_len, target_len


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(  # type: ignore[misc]
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rtc_prefix_physical: np.ndarray | None = None,
        rtc_prefix_len: int | None = None,
        rtc_guidance_weight: float = 5.0,
        rtc_decay_steps: int = 4,
        rtc_schedule: str = "exp",
    ) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # RTC: inject the physical committed-action prefix into `actions` BEFORE
        # the input transform, so the exact same DeltaActions + Normalize used in
        # training convert it to normalized-delta space (anchored on the CURRENT
        # observation state). This makes the prefix live in the same space the
        # flow sampler runs in, with no manual delta/quantile math here.
        use_rtc = (
            rtc_prefix_physical is not None
            and rtc_prefix_len is not None
            and int(rtc_prefix_len) > 0
            and not self._is_pytorch_model
        )
        if use_rtc:
            # Force a writable copy: the prefix arrives via msgpack as a
            # read-only buffer, and DeltaActions modifies `actions` in place.
            inputs["actions"] = np.array(rtc_prefix_physical, dtype=np.float32, copy=True)

        inputs = self._input_transform(inputs)

        # Pull the normalized-delta prefix back out (the transform put it under
        # "actions") and remove it so it is not fed to the model as an input.
        prefix_norm = None
        if use_rtc:
            prefix_norm = np.asarray(inputs.pop("actions"))  # (L, ad) normalized delta

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if use_rtc and prefix_norm is not None:
            ah = self._model.action_horizon
            ad = self._model.action_dim
            # Preserve the COMPLETE normalized remaining chunk as the RTC
            # target. ``lock_len`` controls only the hard-lock portion; later
            # real target rows are available to the decay band.
            prefix_full, weights, _, _ = _prepare_rtc_prefix(
                prefix_norm,
                action_horizon=ah,
                action_dim=ad,
                lock_len=int(rtc_prefix_len),
                decay_steps=rtc_decay_steps,
                schedule=rtc_schedule,
            )
            sample_kwargs["prefix_actions"] = jnp.asarray(prefix_full)[None, ...]
            sample_kwargs["prefix_weights"] = jnp.asarray(weights)
            sample_kwargs["guidance_weight"] = float(rtc_guidance_weight)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        # RTC alignment diagnostic: how well the returned chunk's first
        # prefix_len steps match the physical committed prefix the client sent.
        # Large values here (esp. at the last aligned step vs first free step)
        # explain chunk-boundary jitter. Enable with env RTC_DEBUG=1.
        if use_rtc and os.environ.get("RTC_DEBUG", "0") == "1":
            try:
                out_act = np.asarray(outputs["actions"], dtype=np.float32)
                pl = min(int(rtc_prefix_len), out_act.shape[0], rtc_prefix_physical.shape[0])
                phys_prefix = np.asarray(rtc_prefix_physical, dtype=np.float32)[:pl]
                diff = np.linalg.norm(out_act[:pl] - phys_prefix, axis=-1)
                # boundary: last aligned step vs first free step of the returned chunk
                b = ""
                if out_act.shape[0] > pl:
                    jump = np.linalg.norm(out_act[pl] - out_act[pl - 1])
                    b = f" boundary_jump(step{pl - 1}->{pl})={jump:.4f}"
                print(
                    f"[RTC-align] lock_len={pl} remaining_len={min(rtc_prefix_physical.shape[0], out_act.shape[0])} "
                    f"align_L2 per-step[:4]="
                    f"{np.round(diff[:4], 4).tolist()} mean={diff.mean():.4f}{b}",
                    flush=True,
                )
            except Exception as e:  # never break inference on a diagnostic
                print(f"[RTC-align] diag error: {e}", flush=True)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
