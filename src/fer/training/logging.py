from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Thin TensorBoard wrapper.

    Writes per-epoch scalars (loss, acc, macro_f1, balanced_accuracy) and
    per-class recall under per-class/recal.<cls> so reviewers can monitor
    rare-class behaviour (Disgust, Fear) directly in TensorBoard.
    """

    def __init__(self, log_dir: str | Path):
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_epoch(self, epoch: int, train_loss: float, val: dict) -> None:
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val["loss"], epoch)
        self.writer.add_scalar("metrics/accuracy", val["accuracy"], epoch)
        self.writer.add_scalar("metrics/macro_f1", val["macro_f1"], epoch)
        self.writer.add_scalar("metrics/balanced_accuracy", val["balanced_accuracy"], epoch)
        for i, r in enumerate(val["per_class_recall"]):
            self.writer.add_scalar(f"per_class/recall_{i}", r, epoch)

    def log_hparams(self, hparams: dict, metrics: dict) -> None:
        self.writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
