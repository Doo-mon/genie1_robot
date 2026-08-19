"""Inference config for the G1 3017/3040 H50 joint checkpoint."""

from __future__ import annotations

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.g1_joint_policy as g1_joint_policy
import openpi.transforms as _transforms


def get_g1_joint_configs():
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import DataConfigFactory
    from openpi.training.config import ModelTransformFactory
    from openpi.training.config import TrainConfig

    @dataclasses.dataclass(frozen=True)
    class G1JointH50DataConfig(DataConfigFactory):
        @override
        def create(
            self,
            assets_dirs: pathlib.Path,
            model_config: _model.BaseModelConfig,
        ) -> DataConfig:
            repack = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "observation.images.head_color",
                            "observation/wrist_image": "observation.images.hand_left",
                            "observation/second_image": "observation.images.hand_right",
                            "observation/state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            # Arms are trained as deltas; grippers (indices 7 and 15) remain
            # absolute. Output transforms restore physical absolute joint16.
            joint_delta_mask = _transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = _transforms.Group(
                inputs=[
                    g1_joint_policy.G1JointInputs(
                        model_type=model_config.model_type
                    )
                ],
                outputs=[g1_joint_policy.G1JointOutputs()],
            ).push(
                inputs=[_transforms.DeltaActions(joint_delta_mask)],
                outputs=[_transforms.AbsoluteActions(joint_delta_mask)],
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
            )

    return [
        TrainConfig(
            name="pi05_g1_desk_full_3017_3040_h50_joint",
            model=pi0_config.Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_horizon=50,
            ),
            data=G1JointH50DataConfig(
                repo_id="g1_full_pi_3017_3040_ee_porcess",
                assets=AssetsConfig(asset_id="g1_full_pi_3017_3040_ee_porcess"),
                base_config=DataConfig(prompt_from_task=True),
            ),
        )
    ]
