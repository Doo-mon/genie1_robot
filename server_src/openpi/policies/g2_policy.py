import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_image_with_aliases(data: dict, primary_key: str, aliases: list):
    """Try primary_key first, then fall back to alias keys. Useful when different
    clients (G1 robot vs eval scripts) use slightly different naming."""
    if primary_key in data:
        return data[primary_key]
    for k in aliases:
        if k in data:
            print(f"[G2Inputs] using alias '{k}' for primary='{primary_key}'", flush=True)
            return data[k]
    available = sorted([k for k in data.keys() if "image" in k.lower() or "color" in k.lower()])
    raise KeyError(
        f"Cannot find image for primary_key='{primary_key}'. "
        f"Tried aliases={aliases}. "
        f"Available image-like keys in obs: {available}. "
        f"All keys: {sorted(data.keys())}"
    )


# ---------------------------------------------------------------------------
# Legacy <-> training order remapping (16-d only)
#
#   Legacy client (pi05_g1_desk_macro_joint_h30 protocol):
#       [L_arm7, R_arm7, L_gripper, R_gripper]
#       = idx [0..6, 7..13, 14, 15]
#   Training order (pi05_g1_desk_raw_subtask_h30 -- raw-desk_organization
#   first 16 dims of the 30-d state/action):
#       [L_arm7, L_gripper, R_arm7, R_gripper]
#       = idx [0..6, 7, 8..14, 15]
# ---------------------------------------------------------------------------

# Permutation: out[i] = inp[_LEGACY_TO_TRAINING[i]]
# out (training): [L7@0..6, Lg@7, R7@8..14, Rg@15]
# inp (legacy):   [L7@0..6, R7@7..13, Lg@14, Rg@15]
_LEGACY_TO_TRAINING = [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]
# Inverse permutation: out (legacy) <- inp (training)
_TRAINING_TO_LEGACY = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 7, 15]


def _reorder_legacy_to_training(arr: np.ndarray) -> np.ndarray:
    """Reorder last axis from legacy [L7,R7,Lg,Rg] to training [L7,Lg,R7,Rg].
    Works on shape (16,) and (..., 16)."""
    arr = np.asarray(arr)
    if arr.shape[-1] != 16:
        # Don't touch arrays we don't recognize (e.g. 14-d / 30-d)
        return arr
    return arr[..., _LEGACY_TO_TRAINING]


def _reorder_training_to_legacy(arr: np.ndarray) -> np.ndarray:
    """Reorder last axis from training [L7,Lg,R7,Rg] back to legacy [L7,R7,Lg,Rg]."""
    arr = np.asarray(arr)
    if arr.shape[-1] != 16:
        return arr
    return arr[..., _TRAINING_TO_LEGACY]


@dataclasses.dataclass(frozen=True)
class G2Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    state_key:str = "observation/state"
    action_key:str = "actions"
    image1_key:str = "observation/image"
    # image2_key:str = "observation/wrist_image"
    # image3_key:str = "observation/second_image"
    image2_key:str = "observation/wrist_left_image"
    image3_key:str = "observation/wrist_right_image"
    prompt_key:str = "prompt"
    # When True, the incoming client state is in legacy order [L7, R7, Lg, Rg]
    # (used by old G1 clients designed for the macro_joint 16-d checkpoint),
    # but the model was trained on order [L7, Lg, R7, Rg] (raw-desk_organization
    # 30-d state, first 16 dims). We reorder client -> model order on input,
    # and G2Outputs reorders model -> client order on output.
    # Default False to keep all existing scripts/checkpoints unchanged.
    reorder_legacy_state_action: bool = False

    def __call__(self, data: dict) -> dict:

        ### 先经过config映射  然后再经过policy的映射
        # Support multiple naming conventions from different clients (G1 robot, eval, etc.)
        image1 = _parse_image(_get_image_with_aliases(data, self.image1_key, [
            "observation/image", "observation/head_image", "observation/head_color",
            "observation.images.head_color",
        ]))
        image2 = _parse_image(_get_image_with_aliases(data, self.image2_key, [
            "observation/wrist_left_image", "observation/left_wrist_image",
            "observation/wrist_left", "observation/hand_left_image",
            "observation/hand_left_color", "observation.images.hand_left_color",
            "observation/wrist_image",  # G1 client (legacy macro_joint protocol): wrist_image = left
        ]))
        image3 = _parse_image(_get_image_with_aliases(data, self.image3_key, [
            "observation/wrist_right_image", "observation/right_wrist_image",
            "observation/wrist_right", "observation/hand_right_image",
            "observation/hand_right_color", "observation.images.hand_right_color",
            "observation/second_image",  # G1 client (legacy macro_joint protocol): second_image = right
        ]))

        # State key alias fallback (some clients use "observation/qpos" etc.)
        state_val = data.get(self.state_key)
        if state_val is None:
            for alt in ["observation/state", "observation/qpos", "state", "observation.state"]:
                if alt in data:
                    print(f"[G2Inputs] using alias '{alt}' for state_key='{self.state_key}'", flush=True)
                    state_val = data[alt]
                    break
        if state_val is None:
            raise KeyError(f"Cannot find state. state_key='{self.state_key}'. All keys: {sorted(data.keys())}")

        # Legacy client-order [L7, R7, Lg, Rg] -> training-order [L7, Lg, R7, Rg]
        if self.reorder_legacy_state_action:
            state_val = _reorder_legacy_to_training(state_val)

        # TODO: 这里左边对应的映射最好不要动 否则要改pi模型里面的代码 || 右边的映射由对应的 config 里面指定
        inputs = {
            "state": state_val,
            "image": {
                "base_0_rgb": image1,
                "left_wrist_0_rgb": image2,
                "right_wrist_0_rgb": image3,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if self.action_key in data:
            actions_in = data[self.action_key]
            if self.reorder_legacy_state_action:
                actions_in = _reorder_legacy_to_training(np.asarray(actions_in))
            inputs["actions"] = actions_in

            # # 取第14维之后
            # tail = actions[..., 14:]

            # # 检测异常
            # bad_mask = (tail > 0.9) | (tail < -0.9)

            # if bad_mask.any():
            #     idx = bad_mask.nonzero(as_tuple=False)[0]

            #     print("===== BAD ACTION DETECTED =====")
            #     print("index:", idx.tolist())
            #     print("bad value:", tail[tuple(idx.tolist())].item())

            #     # 打印对应的完整 action
            #     if actions.ndim == 2:
            #         print("full action:", actions[idx[0]])
            #     elif actions.ndim == 3:
            #         print("full action:", actions[idx[0], idx[1]])

            #     print("================================")

            # inputs["actions"] = actions

        # if self.action_key in data:
        #     #actions = data[self.action_key]

        #     #actions = actions.clone()
        #     #actions[..., 14:] = 0
        #     #inputs["actions"] = actions
        #     #print(actions)

        #     # actions[..., 14:]
        #     inputs["actions"] = data[self.action_key]

        if self.prompt_key in data:
            inputs["prompt"] = data[self.prompt_key]

        return inputs


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    action_dim:int = 16
    action_key:str = "actions"
    # When True, reorder model-output actions from training order [L7,Lg,R7,Rg]
    # back to legacy client order [L7,R7,Lg,Rg]. Must match the inverse of
    # G2Inputs.reorder_legacy_state_action.
    reorder_legacy_state_action: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data[self.action_key][:, :self.action_dim])
        if self.reorder_legacy_state_action and actions.shape[-1] == 16:
            actions = _reorder_training_to_legacy(actions)
        return {self.action_key: actions}
