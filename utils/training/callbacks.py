"""Training callbacks for experiment tracking."""

import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class BestModelTracker(TrainerCallback):
    """Tracks the best model epoch and evaluation loss during training."""

    def __init__(self):
        self.best_epoch = None
        self.best_eval_loss = float("inf")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", float("inf"))
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_epoch = state.epoch
            logger.info(
                "New best model at epoch %.2f with eval_loss %.4f",
                self.best_epoch,
                self.best_eval_loss,
            )
