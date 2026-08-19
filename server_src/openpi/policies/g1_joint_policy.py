"""Transforms for G1 16-D joint policies trained from 30-D desk records.

The source records use ``[joint16, ee14]`` while this policy consumes and
predicts only ``[left_arm7, left_gripper, right_arm7, right_gripper]``.
RTC prefixes are already physical joint16 and are accepted unchanged.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


WIRE_DIM = 30
JOINT_DIM = 16
JOINT_SLICE = slice(0, JOINT_DIM)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _extract_joint(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[-1] == WIRE_DIM:
        return array[..., JOINT_SLICE]
    if array.shape[-1] == JOINT_DIM:
        return array
    raise ValueError(
        f"{name} must end in {WIRE_DIM} ([joint16, ee14]) or "
        f"{JOINT_DIM} (joint16), got {array.shape}"
    )


@dataclasses.dataclass(frozen=True)
class G1JointInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    state_key: str = "observation/state"
    action_key: str = "actions"

    def __call__(self, data: dict) -> dict:
        image_keys = (
            "observation/image",
            "observation/wrist_image",
            "observation/second_image",
        )
        required = (*image_keys, self.state_key)
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                f"G1 joint input is missing {missing}; available={sorted(data.keys())}"
            )

        result = {
            "state": _extract_joint(data[self.state_key], name=self.state_key),
            "image": {
                "base_0_rgb": _parse_image(data[image_keys[0]]),
                "left_wrist_0_rgb": _parse_image(data[image_keys[1]]),
                "right_wrist_0_rgb": _parse_image(data[image_keys[2]]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if self.action_key in data:
            result["actions"] = _extract_joint(
                data[self.action_key], name=self.action_key
            )
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class G1JointOutputs(transforms.DataTransformFn):
    action_key: str = "actions"

    def __call__(self, data: dict) -> dict:
        if self.action_key not in data:
            raise KeyError(
                f"G1 joint output is missing {self.action_key}; "
                f"available={sorted(data.keys())}"
            )
        actions = np.asarray(data[self.action_key])
        if actions.ndim != 2 or actions.shape[-1] < JOINT_DIM:
            raise ValueError(f"Expected action chunk (H, >=16), got {actions.shape}")
        return {self.action_key: actions[:, :JOINT_DIM]}
