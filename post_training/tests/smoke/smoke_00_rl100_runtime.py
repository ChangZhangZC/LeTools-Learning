"""Check RL-100 submodule imports and synthetic three-camera RL1002D forward.

Run from the LeTools-Learning checkout after initializing third_party/rl100.
This test uses no pretrained weights and does not require robot data.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RL100_ROOT = REPO_ROOT / "third_party" / "rl100" / "RL-100"
# Batch 00 tests only the 2D Actor path. PPO/IQL/Dynamics will be tested
# independently in later batches. RL1002D itself still has indirect imports.
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
            f"RL-100 submodule not initialized at {RL100_ROOT}. "
            "Run: git submodule update --init third_party/rl100"
        )
    sys.path.insert(0, str(RL100_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "post_training" / "src"))
    print(f"Python: {sys.version.split()[0]}")
    print(f"RL-100 root: {RL100_ROOT}")
    for name in MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if RL100_ROOT.resolve() not in path.parents:
            raise RuntimeError(f"Wrong RL-100 import location for {name}: {path}")
        print(f"[PASS] import {name}")


def check_forward(device_name: str) -> None:
    import torch

    from arbim_rl100.data.batch_adapter import adapt_rl100_shape_meta
    from rl_100.model.vision.model_getter import get_resnet
    from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from rl_100.policy.rl100_2d import RL1002D
    from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (
        DDIMSchedulerExtended,
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    torch.manual_seed(123)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    # Synthetic test shapes, NOT the real Kuavo dataset/model configuration.
    shape_meta = {
        "obs": {
            "rgb_head": {"shape": [3, 84, 84], "type": "rgb"},
            "rgb_left_hand": {"shape": [3, 84, 84], "type": "rgb"},
            "rgb_right_hand": {"shape": [3, 84, 84], "type": "rgb"},
            "agent_pos": {"shape": [16], "type": "low_dim"},
        },
        "action": {"shape": [16]},
    }
    shape_meta = adapt_rl100_shape_meta(shape_meta)
    obs_encoder = MultiImageObsEncoder(
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
        obs_encoder=obs_encoder,
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
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError(f"Non-finite BC loss: {loss}")
    loss.backward()
    for component, parameters in (
        ("U-Net", policy.model.parameters()),
        ("3-RGB encoder", policy.obs_encoder.parameters()),
    ):
        grads = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise AssertionError(f"No finite {component} parameter gradients")
        print(f"[PASS] {component} backward")
    print(f"[PASS] BC loss: {loss.item():.6f}; diagnostics={diagnostics}")

    policy.eval()
    with torch.no_grad():
        action = policy.predict_action(obs, deterministic=True)["action"]
    if tuple(action.shape) != (1, 4, 16) or not torch.isfinite(action).all():
        raise AssertionError(f"Unexpected predicted action: {action.shape}")
    print(f"[PASS] predicted action shape: {tuple(action.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("imports", "forward"), default="imports")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0 (forward only)")
    args = parser.parse_args()
    check_imports()
    if args.level == "forward":
        check_forward(args.device)
    print(f"[PASS] RL-100 {args.level} smoke test")


if __name__ == "__main__":
    main()
