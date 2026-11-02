"""LeRobot processors with native RL-100 normalizer ownership.

No mean/std or min/max from LeRobot is applied twice. The plugin actor
initializes RL-100's own LinearNormalizer from LeRobot dataset metadata.
"""
from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from .configuration_rl100_2d import RL1002DConfig


def make_rl100_2d_pre_post_processors(
    config: RL1002DConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
):
    if any(mode != "IDENTITY" for mode in config.normalization_mapping.values()):
        raise ValueError("RL1002D must use IDENTITY LeRobot processing; native RL-100 normalizes data")
    pre = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    post = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post
