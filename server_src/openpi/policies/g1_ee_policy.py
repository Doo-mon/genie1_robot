"""Transforms for the G1 desk 14-D end-effector policy.

The training dataset stores a 30-D vector in the order ``[joint16, ee14]``.
The model consumes and predicts only the EE suffix.  During RTC inference the
committed prefix is already 14-D, so the input transform deliberately accepts
both representations.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


WIRE_DIM = 30
EE_DIM = 14
EE_SLICE = slice(16, 30)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _extract_ee(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[-1] == WIRE_DIM:
        return array[..., EE_SLICE]
    if array.shape[-1] == EE_DIM:
        # RTC sends the already-extracted physical EE action prefix.
        return array
    raise ValueError(
        f"{name} must end in {WIRE_DIM} ([joint16, ee14]) or {EE_DIM} (ee14), "
        f"got {array.shape}"
    )


@dataclasses.dataclass(frozen=True)
class G1EEInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    state_key: str = "observation/state"
    action_key: str = "actions"
    image1_key: str = "observation/image"
    image2_key: str = "observation/wrist_image"
    image3_key: str = "observation/second_image"
    prompt_key: str = "prompt"

    def __call__(self, data: dict) -> dict:
        required = (self.image1_key, self.image2_key, self.image3_key, self.state_key)
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                f"G1 EE input is missing {missing}; available={sorted(data.keys())}"
            )

        inputs = {
            "state": _extract_ee(data[self.state_key], name=self.state_key),
            "image": {
                "base_0_rgb": _parse_image(data[self.image1_key]),
                "left_wrist_0_rgb": _parse_image(data[self.image2_key]),
                "right_wrist_0_rgb": _parse_image(data[self.image3_key]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if self.action_key in data:
            inputs["actions"] = _extract_ee(
                data[self.action_key], name=self.action_key
            )
        if self.prompt_key in data:
            inputs["prompt"] = data[self.prompt_key]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1EEOutputs(transforms.DataTransformFn):
    action_key: str = "actions"

    def __call__(self, data: dict) -> dict:
        if self.action_key not in data:
            raise KeyError(
                f"G1 EE output is missing {self.action_key}; "
                f"available={sorted(data.keys())}"
            )
        actions = np.asarray(data[self.action_key])
        if actions.ndim != 2 or actions.shape[-1] < EE_DIM:
            raise ValueError(f"Expected action chunk (H, >=14), got {actions.shape}")
        return {self.action_key: actions[:, :EE_DIM]}
