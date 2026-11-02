"""Native RL1002D import and synthetic 3-RGB BC smoke check.

Run from LeTools-Learning root after initializing the RL-100 submodule.
No LeRobot data, pretrained weights, offline RL, or robot is required.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections import UserDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RL100_ROOT = ROOT / "third_party" / "rl100" / "RL-100"
MODULES = (
    "rl_100.policy.rl100_2d",
    "rl_100.model.vision.multi_image_obs_encoder",
    "rl_100.model.diffusion.simple_conditional_unet1d",
    "rl_100.model.diffusion.conditional_unet1d",
    "rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok",
)


def check_imports() -> None:
    if not (RL100_ROOT / "rl_100" / "policy" / "rl100_2d.py").is_file():
        raise FileNotFoundError(
            f"Missing RL-100 checkout at {RL100_ROOT}; "
            "run: git submodule update --init third_party/rl100"
        )
    sys.path.insert(0, str(RL100_ROOT))
    print(f"Python: {sys.version.split()[0]}")
    print(f"RL-100 import root: {RL100_ROOT}")
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__).resolve()
        if RL100_ROOT.resolve() not in module_path.parents:
            raise RuntimeError(f"Wrong RL-100 import for {module_name}: {module_path}")
        print(f"[PASS] import {module_name}")


def check_forward(device_name: str) -> None:
    import torch

    from rl_100.model.vision.model_getter import get_resnet
    from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from rl_100.policy.rl100_2d import RL1002D
    from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (
        DDIMSchedulerExtended,
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(123)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    # Synthetic contract only: NOT the production camera/state/action layout.
    obs_specs = {
        "rgb_head": {"shape": [3, 84, 84], "type": "rgb"},
        "rgb_left_hand": {"shape": [3, 84, 84], "type": "rgb"},
        "rgb_right_hand": {"shape": [3, 84, 84], "type": "rgb"},
        "agent_pos": {"shape": [16], "type": "low_dim"},
    }
    # RL-100 dict_apply recursively descends plain dict metadata values.
    # UserDict keeps each original shape specification as an atomic leaf.
    shape_meta = {
        "obs": {key: UserDict(spec) for key, spec in obs_specs.items()},
        "action": {"shape": [16]},
    }
    encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet("resnet18", weights=None),
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        use_group_norm=False,
        share_rgb_model=True,
        imagenet_norm=True,
        use_agent_pos=True,
        use_vib=False,
        use_recon=False,
    )
    scheduler = DDIMSchedulerExtended(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_std_min=0.01,
    )
    policy = RL1002D(
        shape_meta=shape_meta,
        obs_encoder=encoder,
        cm_noise_scheduler=None,
        ddim_noise_scheduler=scheduler,
        scheduler_type="ddim",
        model="sample_dp3",
        feature_type="2D",
        w_pc=False,
        use_agent_pos=True,
        obs_as_global_cond=True,
        action_norm=False,
        horizon=9,
        n_obs_steps=2,
        n_action_steps=4,
        num_inference_steps=2,
        ddim_inference_steps=2,
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32,
        kernel_size=3,
        n_groups=8,
        use_aug=False,
    ).to(device)

    obs = {
        "rgb_head": torch.rand(1, 2, 3, 84, 84, device=device),
        "rgb_left_hand": torch.rand(1, 2, 3, 84, 84, device=device),
        "rgb_right_hand": torch.rand(1, 2, 3, 84, 84, device=device),
        "agent_pos": torch.zeros(1, 2, 16, device=device),
    }
    batch = {"obs": obs, "action": torch.zeros(1, 9, 16, device=device)}
    policy.train()
    loss, diagnostics = policy.compute_loss(batch)
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise AssertionError(f"Non-finite BC loss: {loss}")
    loss.backward()
    for component, params in (
        ("U-Net", policy.model.parameters()),
        ("3-RGB encoder", policy.obs_encoder.parameters()),
    ):
        grads = [p.grad for p in params if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise AssertionError(f"Missing or non-finite {component} gradients")
        print(f"[PASS] {component} backward")
    print(f"[PASS] BC loss={loss.item():.6f}; metrics={diagnostics}")

    policy.eval()
    with torch.no_grad():
        actions = policy.predict_action(obs, deterministic=True)["action"]
    if tuple(actions.shape) != (1, 4, 16) or not torch.isfinite(actions).all().item():
        raise AssertionError(f"Unexpected predicted actions: {tuple(actions.shape)}")
    print(f"[PASS] predicted actions: {tuple(actions.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("imports", "forward"), default="imports")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    check_imports()
    if args.level == "forward":
        check_forward(args.device)
    print(f"[PASS] RL1002D {args.level}")


if __name__ == "__main__":
    main()
