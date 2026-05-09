"""Fine-tuning utilities for encoder-decoder (seq2seq) models like T5."""

import logging
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig

from utils.training.callbacks import BestModelTracker

logger = logging.getLogger(__name__)


def load_t5_model(
    model_name: str,
) -> tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """Load a T5 model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer


def fine_tune_encoder_decoder(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    save_path: str,
    training_args_kwargs: Optional[dict] = None,
    early_stopping_patience: int = 4,
) -> BestModelTracker:
    """Fine-tune a T5/encoder-decoder model with early stopping.

    Args:
        model_name: HuggingFace model identifier or local path.
        train_dataset: Tokenized training dataset.
        val_dataset: Tokenized validation dataset.
        output_dir: Directory for training checkpoints.
        save_path: Directory to save the final model.
        training_args_kwargs: Override default TrainingArguments parameters.
        early_stopping_patience: Number of epochs without improvement before stopping.

    Returns:
        BestModelTracker with best epoch and eval loss info.
    """
    model, tokenizer = load_t5_model(model_name)

    default_training_args = {
        "output_dir": output_dir,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 12,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 5,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "warmup_steps": 500,
        "gradient_accumulation_steps": 1,
        "disable_tqdm": False,
    }
    if training_args_kwargs:
        default_training_args.update(training_args_kwargs)

    training_args = TrainingArguments(**default_training_args)
    best_model_tracker = BestModelTracker()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            best_model_tracker,
        ],
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    logger.info("Training complete. Model saved to %s", save_path)
    logger.info(
        "Best model at epoch %.2f with eval_loss %.4f",
        best_model_tracker.best_epoch,
        best_model_tracker.best_eval_loss,
    )

    return best_model_tracker
