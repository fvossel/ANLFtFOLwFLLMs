"""Fine-tuning utilities for decoder-only (causal LM) models."""

import logging
from typing import Callable, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel

from utils.training.callbacks import BestModelTracker

logger = logging.getLogger(__name__)


def load_decoder_only_model(
    model_name: str,
    use_cache: bool = False,
    gradient_checkpointing: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a decoder-only model and tokenizer with standard configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_cache=use_cache,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def fine_tune_decoder_only(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    save_path: str,
    lora_config: Optional[LoraConfig] = None,
    training_args_kwargs: Optional[dict] = None,
    gradient_checkpointing: bool = False,
    adapter_path: Optional[str] = None,
    early_stopping_patience: int = 4,
) -> BestModelTracker:
    """Fine-tune a decoder-only model with LoRA and early stopping.

    Args:
        model_name: HuggingFace model identifier.
        train_dataset: Preprocessed training dataset with 'messages' column.
        val_dataset: Preprocessed validation dataset with 'messages' column.
        output_dir: Directory for training checkpoints.
        save_path: Directory to save the final model.
        lora_config: LoRA configuration. If None, uses default causal LM config.
        training_args_kwargs: Override default TrainingArguments parameters.
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency.
        adapter_path: Path to existing LoRA adapter for continued training.
        early_stopping_patience: Number of epochs without improvement before stopping.

    Returns:
        BestModelTracker with best epoch and eval loss info.
    """
    model, tokenizer = load_decoder_only_model(
        model_name, use_cache=False, gradient_checkpointing=gradient_checkpointing
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        model.train()
        logger.info("Loaded existing adapter from %s", adapter_path)

    if lora_config is None:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    default_training_args = {
        "output_dir": output_dir,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": 1e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "gradient_accumulation_steps": 2,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 12,
        "weight_decay": 0.01,
        "bf16": True,
        "save_total_limit": 5,
        "logging_steps": 100,
        "report_to": "none",
    }
    if training_args_kwargs:
        default_training_args.update(training_args_kwargs)

    training_args = TrainingArguments(**default_training_args)
    best_model_tracker = BestModelTracker()

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            best_model_tracker,
        ],
    }
    if not adapter_path:
        trainer_kwargs["peft_config"] = lora_config

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(save_path)

    logger.info("Training complete. Model saved to %s", save_path)
    logger.info(
        "Best model at epoch %.2f with eval_loss %.4f",
        best_model_tracker.best_epoch,
        best_model_tracker.best_eval_loss,
    )

    return best_model_tracker
