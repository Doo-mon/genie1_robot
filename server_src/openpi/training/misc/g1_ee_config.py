"""Inference-compatible configs for Zhanzhihao's G1 desk EE checkpoints."""

from __future__ import annotations

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.g1_ee_policy as g1_ee_policy
import openpi.transforms as _transforms


def get_g1_ee_configs():
    # Local imports avoid the config.py <-> misc config circular dependency.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import DataConfigFactory
    from openpi.training.config import ModelTransformFactory
    from openpi.training.config import TrainConfig

    @dataclasses.dataclass(frozen=True)
    class G1EEH20DataConfig(DataConfigFactory):
        @override
        def create(
            self,
            assets_dirs: pathlib.Path,
            model_config: _model.BaseModelConfig,
        ) -> DataConfig:
            # Repack is used by training/data tools. Policy inference already
            # sends the keys on the left and therefore starts at data_transforms.
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
            data_transforms = _transforms.Group(
                inputs=[g1_ee_policy.G1EEInputs(model_type=model_config.model_type)],
                outputs=[g1_ee_policy.G1EEOutputs()],
            ).push(
                inputs=[
                    _transforms.DeltaActions(
                        _transforms.make_bool_mask(6, -1, 6, -1)
                    )
                ],
                outputs=[
                    _transforms.AbsoluteActions(
                        _transforms.make_bool_mask(6, -1, 6, -1)
                    )
                ],
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
            name="pi05_g1_desk_multitask_h20",
            model=pi0_config.Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_horizon=20,
            ),
            data=G1EEH20DataConfig(
                repo_id="g1_desk_multitask",
                assets=AssetsConfig(asset_id="g1_desk_multitask"),
                base_config=DataConfig(prompt_from_task=True),
            ),
        ),
        TrainConfig(
            name="pi05_g1_desk_multitask_h50_ee_new",
            model=pi0_config.Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_horizon=50,
            ),
            data=G1EEH20DataConfig(
                repo_id="g1_1w_lerobot_episode_unwrapped",
                assets=AssetsConfig(asset_id="g1_1w_lerobot_episode_unwrapped"),
                base_config=DataConfig(prompt_from_task=True),
            ),
        ),
        TrainConfig(
            name="pi05_g1_desk_full_3017_3040_h50_ee",
            model=pi0_config.Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_horizon=50,
            ),
            data=G1EEH20DataConfig(
                repo_id="g1_full_pi_3017_3040",
                assets=AssetsConfig(asset_id="g1_full_pi_3017_3040"),
                base_config=DataConfig(prompt_from_task=True),
            ),
        ),
        TrainConfig(
            name="pi05_g1_desk_full_3017_3040_h50_ee_process",
            model=pi0_config.Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_horizon=50,
            ),
            data=G1EEH20DataConfig(
                repo_id="g1_full_pi_3017_3040_ee_porcess",
                assets=AssetsConfig(asset_id="g1_full_pi_3017_3040_ee_porcess"),
                base_config=DataConfig(prompt_from_task=True),
            ),
        ),
        TrainConfig(
            name="pi05_g1_desk_full_3017_3040_h50_camera_ee",
            model=pi0_config.Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_horizon=50,
            ),
            data=G1EEH20DataConfig(
                repo_id="g1_full_pi_3017_3040_camera_ee",
                assets=AssetsConfig(asset_id="g1_full_pi_3017_3040_camera_ee"),
                base_config=DataConfig(prompt_from_task=True),
            ),
        ),
        TrainConfig(
            name="pi05_g1_desk_full_5659_h50_camera_ee",
            model=pi0_config.Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_horizon=50,
            ),
            data=G1EEH20DataConfig(
                repo_id="g1_full_pi_5659_camera_ee",
                assets=AssetsConfig(asset_id="g1_full_pi_5659_camera_ee"),
                base_config=DataConfig(prompt_from_task=True),
            ),
        ),
    ]
