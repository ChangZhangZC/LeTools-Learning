"""Opt-in compatibility for the pinned RL-100 2D-only import path.

This never modifies the RL-100 submodule. It exists because the pinned
RL1002D source imports pytorch3d.ops but never uses that alias.
Never enable this fallback for a point-cloud / 3D model.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
import warnings
from pathlib import Path


class _DisabledPointCloudOps(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            "PyTorch3D point-cloud operator requested by a 2D-only "
            "compatibility shim. Install real PyTorch3D to use 3D ops."
        )


def enable_rl100_2d_only_import(rl100_root: Path) -> bool:
    """Allow the *unused* pytorch3d.ops import in the pinned RL1002D source.

    Returns True if the missing-dependency shim was enabled, False if the
    real pytorch3d package is available. Refuses to proceed if RL1002D
    actually uses the torch3d_ops alias; this safeguards upstream upgrades.
    """
    if importlib.util.find_spec("pytorch3d") is not None:
        return False

    source = Path(rl100_root) / "rl_100" / "policy" / "rl100_2d.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imports = [
        alias
        for node in ast.walk(parsed)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "pytorch3d.ops"
    ]
    if len(imports) != 1 or imports[0].asname != "torch3d_ops":
        raise RuntimeError(
            "RL1002D PyTorch3D import contract changed; refusing the shim"
        )
    if any(
        isinstance(node, ast.Name) and node.id == "torch3d_ops"
        for node in ast.walk(parsed)
    ):
        raise RuntimeError(
            "RL1002D now uses PyTorch3D operators; install real PyTorch3D"
        )

    package = types.ModuleType("pytorch3d")
    package.__path__ = []
    package.__package__ = "pytorch3d"
    ops = _DisabledPointCloudOps("pytorch3d.ops")
    ops.__package__ = "pytorch3d"
    package.ops = ops
    sys.modules["pytorch3d"] = package
    sys.modules["pytorch3d.ops"] = ops
    warnings.warn(
        "PyTorch3D is unavailable; only the unused RL1002D import is "
        "bypassed. Point-cloud operations are disabled.",
        RuntimeWarning,
        stacklevel=2,
    )
    return True
