from typing import TypeAlias
import dataclasses
import os
import pathlib
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

import openpi.policies.g2_policy as g2_policy

ModelType: TypeAlias = _model.ModelType

def _resolve_dataset_root(env_name: str, candidates: tuple[str, ...]) -> str:
    override = os.environ.get(env_name)
    if override:
        return override
    for candidate in candidates:
        if (pathlib.Path(candidate) / "meta" / "info.json").exists():
            return candidate
    return candidates[0]


G1_DESK_DATA_ROOT = _resolve_dataset_root(
    "G1_DESK_DATA_ROOT",
    (
        "/mnt/workspace-20260313/datasets/suqian_agibot_kingkong/desk_organization_pi05_macro_joint",
        "/mnt/workspace1/datasets/suqian_agibot_kingkong/desk_organization_pi05_macro_joint",
    ),
)
G1_DESK_SMOKE_DATA_ROOT = _resolve_dataset_root(
    "G1_DESK_SMOKE_DATA_ROOT",
    (
        "/mnt/workspace-20260313/datasets/suqian_agibot_kingkong/desk_organization_pi05_macro_joint_smoke",
        "/mnt/workspace1/datasets/suqian_agibot_kingkong/desk_organization_pi05_macro_joint_smoke",
    ),
)


def get_g2_configs():

    from openpi.training.config import AssetsConfig
    from openpi.training.config import TrainConfig
    from openpi.training.config import DataConfigFactory, DataConfig, ModelTransformFactory

    @dataclasses.dataclass(frozen=True)
    class G2DataConfig(DataConfigFactory):
        extra_delta_transform: bool = False
        # Mask for dimensions converted to deltas. Positive spans are joint
        # dimensions; negative spans are absolute dimensions such as grippers.
        delta_action_mask: tuple[bool, ...] | None = None

        action_dim:int = 16
        repo_root: str | None = None
        macro_boundary_action: bool = False

        action_key:str = "actions"
        state_key:str = "observation/state"
        image1_key:str = "observation/image"
        image2_key:str = "observation/wrist_left_image"
        image3_key:str = "observation/wrist_right_image"
        prompt_key:str = "prompt"
        image1_source_key: str = "observation.images.head_color"
        image2_source_key: str = "observation.images.hand_left_color"
        image3_source_key: str = "observation.images.hand_right_color"
        # When True, runtime policy.infer() will reorder the incoming client
        # state (legacy [L7,R7,Lg,Rg]) into training order [L7,Lg,R7,Rg] and
        # reorder model-output actions back to legacy order. Only set this for
        # the raw-desk_organization (raw_subtask) checkpoint. Leave False for
        # the older macro_joint checkpoint and for training.
        reorder_legacy_state_action: bool = False


        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            self.image1_key: self.image1_source_key,
                            self.image2_key: self.image2_source_key,
                            self.image3_key: self.image3_source_key,
                            self.state_key: "state",
                            self.action_key: "actions",
                            self.prompt_key: "prompt",
                        })]
            )
            data_transforms = _transforms.Group(
                inputs=[g2_policy.G2Inputs(model_type=model_config.model_type,
                                           state_key=self.state_key,
                                           action_key=self.action_key,
                                           image1_key=self.image1_key,
                                           image2_key=self.image2_key,
                                           image3_key=self.image3_key,
                                           prompt_key=self.prompt_key,
                                           reorder_legacy_state_action=self.reorder_legacy_state_action,)],
                outputs=[g2_policy.G2Outputs(action_dim=self.action_dim,
                                             action_key=self.action_key,
                                             reorder_legacy_state_action=self.reorder_legacy_state_action,)],
            )
            if self.extra_delta_transform:
                delta_action_mask = self.delta_action_mask or _transforms.make_bool_mask(14, -2)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )
            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                repo_root=self.repo_root,
                macro_boundary_action=self.macro_boundary_action,
                macro_action_key=self.action_key,
            )



    return [


        TrainConfig(
            name="pi05_g1_desk_macro_joint_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="desk_organization_pi05_macro_joint",
                repo_root=G1_DESK_DATA_ROOT,
                macro_boundary_action=True,
                extra_delta_transform=True,
                action_dim=16,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),


        TrainConfig(
            name="pi05_g1_desk_macro_joint_h30_smoke",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=10,
                peak_lr=3e-5,
                decay_steps=100,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="desk_organization_pi05_macro_joint_smoke",
                repo_root=G1_DESK_SMOKE_DATA_ROOT,
                macro_boundary_action=True,
                extra_delta_transform=True,
                action_dim=16,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi/checkpoints",
            num_train_steps=2,
            batch_size=2,
            num_workers=0,
            log_interval=1,
            save_interval=1,
            keep_period=1,
        ),


        TrainConfig(
            name="pi05_g2_cup_righteye_3960_4007_4074_4235_subtask",
            model=pi0_config.Pi0Config(pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data =G2DataConfig(
                repo_id="G2_cup_righteye_3960_4007_4074_4235_subtask",
                extra_delta_transform=True,
                action_dim=16,
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
                base_config = DataConfig(prompt_from_task=True,),
            ),
            num_train_steps=30000,
            batch_size=128,
            log_interval=100,
        ),








        TrainConfig(
            name="pi05_g2_3799",
            model=pi0_config.Pi0Config(pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data =G2DataConfig(
                repo_id="G2_cup_fisheye_3799",  # TODO: 这个构建数据集的时候 名字命名错了 将就用吧
                extra_delta_transform=True,
                action_dim=16,
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
                base_config = DataConfig(prompt_from_task=True,),
            ),
            num_train_steps=30000,
            batch_size=2,
            log_interval=10,
        ),

        TrainConfig(
            name="pi05_g2_3799_exclude",
            model=pi0_config.Pi0Config(pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data =G2DataConfig(
                repo_id="G2_3799_exclude",
                extra_delta_transform=True,
                action_dim=16,
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
                base_config = DataConfig(prompt_from_task=True,),
            ),
            num_train_steps=30000,
            batch_size=32,
            log_interval=10,
        ),




        ## TODO: 神奇的是这边开了 delta  但是训练也是正常的
        TrainConfig(
            name="pi05_g2_cup_3960",
            model=pi0_config.Pi0Config(action_horizon=10, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data =G2DataConfig(
                repo_id="G2_cup_fisheye_3960",
                extra_delta_transform=True,
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
                base_config = DataConfig(prompt_from_task=True,),
            ),
            num_train_steps=30000,
            batch_size=32,
            log_interval=10,
        ),





        TrainConfig(
            name="pi05_g2_cup_4007",
            model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data =G2DataConfig(
                repo_id="G2_cup_fisheye_4007",
                extra_delta_transform=True,
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
                base_config = DataConfig(prompt_from_task=True,),
            ),
            num_train_steps=30000,
            batch_size=32,
            log_interval=10,
        ),



        TrainConfig(
            name="pi05_g2_cup_fisheye",
            model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data =G2DataConfig(
                repo_id="G2_cup_fisheye",
                extra_delta_transform=True,
                base_config = DataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets",),
            ),
            num_train_steps=30000,
            batch_size=32,
            log_interval=10,
        ),



        ####### TODO: 这个完全有问题
        TrainConfig(
            name="pi05_g2_cup_fisheye_abs",
            model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data =G2DataConfig(
                repo_id="G2_cup_fisheye",
                extra_delta_transform=False,
                base_config = DataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets", asset_id="G2_cup_fisheye"),
            ),
            num_train_steps=30000,
            batch_size=32,
            log_interval=10,
        ),






        #################### 这个小样本测试的代码  在开了detla的情况 loss异常

        # TrainConfig(
        #     name="pi05_g2_cup_base",
        #     model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
        #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        #     ema_decay=0.999,
        #     data =G2DataConfig(
        #         repo_id="G2_cup_fisheye_test",
        #         extra_delta_transform=False,
        #         assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
        #         base_config = DataConfig(prompt_from_task=True,),
        #     ),
        #     num_train_steps=30000,
        #     batch_size=32,
        #     log_interval=10,
        # ),



        # ### 这个loss不正常
        # TrainConfig(
        #     name="pi05_g2_cup_delta",
        #     model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
        #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        #     ema_decay=0.999,
        #     data =G2DataConfig(
        #         repo_id="G2_cup_fisheye_test",
        #         extra_delta_transform=True, # TODO:
        #         assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
        #         base_config = DataConfig(prompt_from_task=True,),
        #     ),
        #     num_train_steps=30000,
        #     batch_size=32,
        #     log_interval=10,
        # ),


        # TrainConfig(
        #     name="pi05_g2_cup_h16",
        #     model=pi0_config.Pi0Config(action_horizon=16, pi05=True),
        #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        #     ema_decay=0.999,
        #     data =G2DataConfig(
        #         repo_id="G2_cup_fisheye_test",
        #         assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets"),
        #         base_config = DataConfig(prompt_from_task=True,),
        #     ),
        #     num_train_steps=30000,
        #     batch_size=32,
        #     log_interval=10,
        # ),






        ######################### 这个是旧的并行处理的代码
        # TrainConfig(
        #     name="pi05_g2_cup",
        #     model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
        #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        #     ema_decay=0.999,
        #     data =G2DataConfig(
        #         repo_id="G2_cup",
        #         base_config = DataConfig(prompt_from_task=True,),
        #         assets=AssetsConfig(assets_dir="/mnt/workspace/users/zhanzhihao/openpi/assets", # TODO:
        #                             asset_id="G2_cup_fisheye_test"),
        #     ),
        #     num_train_steps=30000,
        #     batch_size=32,
        #     log_interval=10,
        # ),


        # TrainConfig(
        #     name="pi05_g2",
        #     model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
        #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        #     ema_decay=0.999,
        #     data =G2DataConfig(
        #         repo_id="G2",
        #         base_config = DataConfig(prompt_from_task=True,),
        #     ),
        #     num_train_steps=30000,
        #     batch_size=32,
        # ),


        # TrainConfig(
        #     name="pi05_g2_new",
        #     model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
        #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        #     ema_decay=0.999,
        #     data =G2DataConfig(
        #         repo_id="G2_new",
        #         base_config = DataConfig(prompt_from_task=True,),
        #     ),
        #     num_train_steps=30000,
        #     batch_size=32,
        # ),



        # TrainConfig(
        #     name="pi05_g2_test",
        #     model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
        #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        #     ema_decay=0.999,
        #     data =G2DataConfig(
        #         repo_id="G2_new_test",
        #         base_config = DataConfig(prompt_from_task=True,),
        #     ),
        #     num_train_steps=30000,
        #     batch_size=32,
        # ),


        # =====================================================================
        # raw-desk_organization: pi0.5 multi-subtask post-train, online subtask
        # boundary, no parquet preprocessing.
        # =====================================================================
        TrainConfig(
            name="pi05_g1_desk_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="raw-desk_organization",
                repo_root=os.environ.get(
                    "RAW_DESK_DATA_ROOT",
                    "/mnt/workspace/users/limingyang/datasets/raw-desk_organization",
                ),
                macro_boundary_action=False,
                extra_delta_transform=True,
                action_dim=16,
                # The G1 robot client sends state in legacy order [L7,R7,Lg,Rg]
                # (designed for the older macro_joint 16-d checkpoint), but
                # raw-desk_organization training data uses [L7,Lg,R7,Rg] for
                # the first 16 dims. Enable runtime reordering on the policy
                # server so the client doesn't need to change. Training/data
                # loader is unaffected (this only kicks in at policy.infer()).
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get(
                        "RAW_DESK_DATA_ROOT",
                        "/mnt/workspace/users/limingyang/datasets/raw-desk_organization",
                    ),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),

        # =====================================================================
        # G1 tea-table (茶几) single-task post-train configs.
        # 6 tasks, each trained independently on pi05_base with raw_subtask
        # boundary mode. No merge needed -- each task gets its own config,
        # norm stats, and checkpoint.
        # =====================================================================
        TrainConfig(
            name="pi05_g1_teatable_task5105_raw_subtask_h30",
            project_name="g1_kingkong_teatable",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="G1_task_5105_subtask_new_eepose",
                repo_root=os.environ.get(
                    "G1_TEATABLE_5105_DATA_ROOT",
                    "/mnt/workspace/users/limingyang/datasets/G1_task_5105_subtask_new_eepose",
                ),
                macro_boundary_action=False,
                extra_delta_transform=True,
                action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get(
                        "G1_TEATABLE_5105_DATA_ROOT",
                        "/mnt/workspace/users/limingyang/datasets/G1_task_5105_subtask_new_eepose",
                    ),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_teatable_task5137_raw_subtask_h30",
            project_name="g1_kingkong_teatable",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="G1_task_5137_subtask_new_eepose",
                repo_root=os.environ.get(
                    "G1_TEATABLE_5137_DATA_ROOT",
                    "/mnt/workspace/users/limingyang/datasets/G1_task_5137_subtask_new_eepose",
                ),
                macro_boundary_action=False,
                extra_delta_transform=True,
                action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get(
                        "G1_TEATABLE_5137_DATA_ROOT",
                        "/mnt/workspace/users/limingyang/datasets/G1_task_5137_subtask_new_eepose",
                    ),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_teatable_task5306_raw_subtask_h30",
            project_name="g1_kingkong_teatable",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="G1_task_5306_subtask_new_eepose",
                repo_root=os.environ.get(
                    "G1_TEATABLE_5306_DATA_ROOT",
                    "/mnt/workspace/users/limingyang/datasets/G1_task_5306_subtask_new_eepose",
                ),
                macro_boundary_action=False,
                extra_delta_transform=True,
                action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get(
                        "G1_TEATABLE_5306_DATA_ROOT",
                        "/mnt/workspace/users/limingyang/datasets/G1_task_5306_subtask_new_eepose",
                    ),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_teatable_task5360_raw_subtask_h30",
            project_name="g1_kingkong_teatable",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="G1_task_5360_subtask_new_eepose",
                repo_root=os.environ.get(
                    "G1_TEATABLE_5360_DATA_ROOT",
                    "/mnt/workspace/users/limingyang/datasets/G1_task_5360_subtask_new_eepose",
                ),
                macro_boundary_action=False,
                extra_delta_transform=True,
                action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get(
                        "G1_TEATABLE_5360_DATA_ROOT",
                        "/mnt/workspace/users/limingyang/datasets/G1_task_5360_subtask_new_eepose",
                    ),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_teatable_task5375_raw_subtask_h30",
            project_name="g1_kingkong_teatable",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="G1_task_5375_subtask_new_eepose",
                repo_root=os.environ.get(
                    "G1_TEATABLE_5375_DATA_ROOT",
                    "/mnt/workspace/users/limingyang/datasets/G1_task_5375_subtask_new_eepose",
                ),
                macro_boundary_action=False,
                extra_delta_transform=True,
                action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get(
                        "G1_TEATABLE_5375_DATA_ROOT",
                        "/mnt/workspace/users/limingyang/datasets/G1_task_5375_subtask_new_eepose",
                    ),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_teatable_task5390_raw_subtask_h30",
            project_name="g1_kingkong_teatable",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2000,
                peak_lr=3e-5,
                decay_steps=300000,
                decay_lr=3e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="G1_task_5390_subtask_new_eepose",
                repo_root=os.environ.get(
                    "G1_TEATABLE_5390_DATA_ROOT",
                    "/mnt/workspace/users/limingyang/datasets/G1_task_5390_subtask_new_eepose",
                ),
                macro_boundary_action=False,
                extra_delta_transform=True,
                action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get(
                        "G1_TEATABLE_5390_DATA_ROOT",
                        "/mnt/workspace/users/limingyang/datasets/G1_task_5390_subtask_new_eepose",
                    ),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=5000,
            keep_period=5000,
        ),

        # =====================================================================
        # G1 desk single-task (8 sub-tasks) post-train configs.
        # Each task trained independently on pi05_base with raw_subtask
        # boundary mode. Data in /mnt/workspace1/datasets/G1_desk_all/.
        # =====================================================================
        TrainConfig(
            name="pi05_g1_desk_books_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_books",
                repo_root=os.environ.get("G1_DESK_BOOKS_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_books"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_BOOKS_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_books"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_desk_charger_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_charger",
                repo_root=os.environ.get("G1_DESK_CHARGER_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_charger"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_CHARGER_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_charger"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_desk_correction_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_correction",
                repo_root=os.environ.get("G1_DESK_CORRECTION_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_correction"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_CORRECTION_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_correction"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_desk_eraser_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_eraser",
                repo_root=os.environ.get("G1_DESK_ERASER_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_eraser"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_ERASER_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_eraser"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_desk_pen_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_pen",
                repo_root=os.environ.get("G1_DESK_PEN_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_pen"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_PEN_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_pen"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_desk_pencil_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_pencil",
                repo_root=os.environ.get("G1_DESK_PENCIL_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_pencil"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_PENCIL_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_pencil"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_desk_stapler_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_stapler",
                repo_root=os.environ.get("G1_DESK_STAPLER_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_stapler"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_STAPLER_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_stapler"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        TrainConfig(
            name="pi05_g1_desk_trash_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_trash",
                repo_root=os.environ.get("G1_DESK_TRASH_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_trash"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_TRASH_DATA_ROOT", "/mnt/workspace/users/limingyang/datasets/g1_trash"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=300000, batch_size=128, num_workers=16, log_interval=50, save_interval=5000, keep_period=5000,
        ),

        # =====================================================================
        # G1 desk ALL merged (8 sub-tasks combined) post-train config.
        # Merges all 8 desk tasks into one dataset for multi-task training.
        # Run scripts/merge_desk_all_datasets.py first to create the merged dataset.
        # =====================================================================
        TrainConfig(
            name="pi05_g1_desk_all_merged_raw_subtask_h30",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_desk_all_merged",
                repo_root=os.environ.get("G1_DESK_ALL_MERGED_DATA_ROOT", "/mnt/workspace1/datasets/G1_desk_merged"),
                macro_boundary_action=False, extra_delta_transform=True, action_dim=16,
                reorder_legacy_state_action=True,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets"),
                base_config=DataConfig(prompt_from_task=False, action_sequence_keys=(), tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_ALL_MERGED_DATA_ROOT", "/mnt/workspace1/datasets/G1_desk_merged"),
                    raw_subtask_action_key="action", raw_subtask_state_key="observation.state", raw_subtask_state_action_dim=16),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=100000, batch_size=128, num_workers=16, log_interval=50, save_interval=10000, keep_period=10000,
        ),

        # Fixed raw-16 layout for the merged desk dataset.
        #
        # Dataset meta/modality.json defines the first 16 dims as:
        #   [left_arm7, left_hand1, right_arm7, right_hand1].
        # Therefore no legacy [L7,R7,Lg,Rg] reorder should be applied, and the
        # delta mask must leave dims 7 and 15 absolute.
        TrainConfig(
            name="pi05_g1_desk_all_merged_raw_subtask_h30_fixed",
            project_name="g1_kingkong_desk",
            model=pi0_config.Pi0Config(action_horizon=30, pi05=True),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=G2DataConfig(
                repo_id="g1_desk_all_merged",
                repo_root=os.environ.get("G1_DESK_ALL_MERGED_DATA_ROOT", "/mnt/workspace1/datasets/G1_desk_merged"),
                macro_boundary_action=False,
                extra_delta_transform=True,
                delta_action_mask=_transforms.make_bool_mask(7, -1, 7, -1),
                action_dim=16,
                reorder_legacy_state_action=False,
                image1_source_key="observation.images.head_color",
                image2_source_key="observation.images.hand_left",
                image3_source_key="observation.images.hand_right",
                assets=AssetsConfig(
                    assets_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
                    asset_id="g1_desk_all_merged_raw16_fixed",
                ),
                base_config=DataConfig(
                    prompt_from_task=False,
                    action_sequence_keys=(),
                    tolerance_s=0.02,
                    raw_subtask_root=os.environ.get("G1_DESK_ALL_MERGED_DATA_ROOT", "/mnt/workspace1/datasets/G1_desk_merged"),
                    raw_subtask_action_key="action",
                    raw_subtask_state_key="observation.state",
                    raw_subtask_state_action_dim=16,
                ),
            ),
            assets_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/assets",
            checkpoint_base_dir="/mnt/workspace/users/limingyang/openpi_raw_subtask/checkpoints",
            num_train_steps=100000,
            batch_size=128,
            num_workers=16,
            log_interval=50,
            save_interval=10000,
            keep_period=10000,
        ),

    ]
