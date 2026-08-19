"""Real-Time Chunking (RTC) policy wrapper — client-prefix variant.

RTC (Kevin Black et al., Physical Intelligence; PI-Kinetix "guidance" method)
makes chunked VLA policies produce smooth, reactive trajectories under async
execution: while the robot is still executing the tail of the previous action
chunk, the next chunk is generated *conditioned on the actions already
committed*, so the two chunks stitch together without a boundary jump.

This variant is aligned with the JoyRA server protocol: the CLIENT sends back
the still-to-be-executed portion of the previous chunk as a physical-space
action prefix, plus how many steps of it are committed:

    obs["remaining_action_chunk"]  # (L, 16) physical joint actions, model order
    obs["prefix_len"]  (or obs["rtc_overlap_steps"])  # hard-lock step count <= L

The wrapper is therefore stateless — it just extracts those fields and forwards
them to `Policy.infer`, which routes the physical prefix through the SAME
DeltaActions + Normalize used in training (anchored on the current observation
state) to obtain the normalized-delta prefix the flow sampler needs. All the
delta/normalization math lives in the transform pipeline, not here.
"""

from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.policies import policy as _policy


def _parse_prefix_len(obs: dict) -> int | None:
    for key in ("rtc_overlap_steps", "prefix_len"):
        v = obs.get(key)
        if v is not None:
            return max(0, int(np.asarray(v).reshape(-1)[0]))
    return None


class RTCPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        base_policy: _policy.Policy,
        *,
        guidance_weight: float = 5.0,
        decay_steps: int = 4,
        schedule: str = "exp",
    ):
        """Wrap a base `Policy` with client-provided RTC prefix guidance.

        Args:
            base_policy: the underlying JAX policy.
            guidance_weight: max guidance-weight cap (PI-Kinetix
                ``rtc_max_guidance_weight``). The per-step weight is computed
                dynamically as a function of denoising time.
            decay_steps: maximum soft-decay length after the locked prefix. The
                band is clipped to the real remaining-action target length.
            schedule: decay-band schedule ("exp", "linear", "ones", "zeros").
        """
        self._base = base_policy
        self._guidance_weight = float(guidance_weight)
        self._decay_steps = int(decay_steps)
        self._schedule = schedule

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        remaining = obs.get("remaining_action_chunk")
        if remaining is not None and np.size(remaining) == 0:
            remaining = None
        prefix_len = _parse_prefix_len(obs)

        kwargs: dict[str, Any] = {}
        if remaining is not None and prefix_len is not None and prefix_len > 0:
            rem = np.asarray(remaining, dtype=np.float32)
            if rem.ndim == 3 and rem.shape[0] == 1:
                rem = rem[0]
            if rem.ndim == 1:
                rem = rem[None, :]
            kwargs = {
                "rtc_prefix_physical": rem,
                "rtc_prefix_len": prefix_len,
                "rtc_guidance_weight": self._guidance_weight,
                "rtc_decay_steps": self._decay_steps,
                "rtc_schedule": self._schedule,
            }

        # Strip RTC-only keys so they never reach the model input transforms.
        obs_clean = {
            k: v for k, v in obs.items() if k not in ("remaining_action_chunk", "prefix_len", "rtc_overlap_steps")
        }
        return self._base.infer(obs_clean, **kwargs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._base.metadata
