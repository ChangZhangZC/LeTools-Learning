"""Optional WandB/TensorBoard sinks for native RL100 DP-2D BC metrics.

metrics.jsonl is the always-on durable record. This adapter has no model,
optimizer, scheduler or RNG involvement: logging never changes BC behavior.
Optional packages are imported only if enabled in the production YAML.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


class BCMetricLogger:
    def __init__(self, cfg, output: Path, *, resumed: bool, resume_step: int):
        self._tb = None
        self._wb_run = None
        self._wb_id_path = output / "wandb_run_id.txt"
        wb = cfg.get("wandb", {}) or {}
        tb = cfg.get("tensorboard", {}) or {}
        if not bool(wb.get("enable", False)) and not bool(tb.get("enable", False)):
            return
        try:
            if bool(wb.get("enable", False)):
                project = str(wb.get("project", "")).strip()
                if not project:
                    raise ValueError("wandb.enable=true requires wandb.project")
                try:
                    import wandb
                except ImportError as exc:
                    raise ImportError(
                        "WandB is enabled. Install with: pip install wandb"
                    ) from exc
                run_id = None
                if resumed:
                    if not self._wb_id_path.is_file():
                        raise FileNotFoundError(
                            "Cannot resume WandB without wandb_run_id.txt in the "
                            "original output directory"
                        )
                    run_id = self._wb_id_path.read_text(encoding="utf-8").strip()
                    if not run_id:
                        raise ValueError("Empty saved WandB run id")
                self._wb_run = wandb.init(
                    project=project,
                    entity=wb.get("entity", None),
                    mode=str(wb.get("mode", "online")),
                    dir=str(output),
                    name=output.name,
                    id=run_id,
                    resume="allow" if resumed else None,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                if self._wb_run is None or not getattr(self._wb_run, "id", None):
                    raise RuntimeError("wandb.init did not return a valid run")
                if not resumed:
                    self._wb_id_path.write_text(
                        str(self._wb_run.id) + "\n", encoding="utf-8"
                    )
                # Use the trainer's optimizer-step axis instead of WandB's
                # internal event counter: train/val may log at the same step.
                self._wb_run.define_metric("trainer_step")
                self._wb_run.define_metric("train/*", step_metric="trainer_step")
                self._wb_run.define_metric("val/*", step_metric="trainer_step")
            if bool(tb.get("enable", False)):
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError as exc:
                    raise ImportError(
                        "TensorBoard is enabled. Install with: pip install tensorboard"
                    ) from exc
                options = {"purge_step": resume_step + 1} if resumed else {}
                self._tb = SummaryWriter(log_dir=str(output / "tensorboard"), **options)
        except BaseException:
            self.close()
            raise

    def _log(self, step: int, values: dict[str, float]) -> None:
        if self._wb_run is not None:
            self._wb_run.log({"trainer_step": step, **values})
        if self._tb is not None:
            for key, val in values.items():
                self._tb.add_scalar(key, val, global_step=step)
            self._tb.flush()

    def train(self, *, step: int, loss: float, lr: float, epoch: float) -> None:
        self._log(step, {
            "train/loss": loss,
            "train/lr": lr,
            "train/epoch_equivalent": epoch,
        })

    def validation(self, *, step: int, loss: float, best_loss: float) -> None:
        self._log(step, {
            "val/loss": loss,
            "val/best_loss": best_loss,
        })

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._wb_run is not None:
            self._wb_run.finish()
            self._wb_run = None
