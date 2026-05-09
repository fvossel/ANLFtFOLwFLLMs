"""Batch inference utilities for decoder-only (causal LM) models."""

import json
import logging
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


def load_decoder_only_for_inference(
    model_name: str,
    adapter_path: Optional[str] = None,
    merge_adapter: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a decoder-only model for inference, optionally with a LoRA adapter.

    Args:
        model_name: HuggingFace model identifier or local path.
        adapter_path: Path to a LoRA adapter to load on top of the base model.
        merge_adapter: Whether to merge the adapter into the base model weights.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_cache=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, device_map="auto", is_trainable=False
        )
        if merge_adapter:
            model = model.merge_and_unload()
        logger.info("Loaded adapter from %s (merged=%s)", adapter_path, merge_adapter)

    return model, tokenizer


def run_batch_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: list[dict],
    formatting_func: Callable,
    output_file: str,
    result_builder: Optional[Callable] = None,
    batch_size: int = 32,
    max_new_tokens: int = 250,
    strip_input: bool = False,
) -> list[dict]:
    """Run batch inference on a decoder-only model.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        data: List of input dictionaries.
        formatting_func: Function that takes an item and returns a prompt string.
        output_file: Path to save results after each batch.
        result_builder: Optional function(item, response) -> dict for custom result format.
            If None, uses default format with NL, FOL_PRED, FOL_GROUND-TRUTH.
        batch_size: Number of examples per batch.
        max_new_tokens: Maximum number of tokens to generate.
        strip_input: If True, only decode the newly generated tokens (exclude input).

    Returns:
        List of result dictionaries.
    """
    results = []

    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        prompts = [formatting_func(item) for item in batch_data]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        if strip_input:
            input_len = inputs["input_ids"].shape[1]
            responses = [
                tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                for output in outputs
            ]
        else:
            responses = [
                tokenizer.decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for output in outputs
            ]

        for idx, item in enumerate(batch_data):
            if result_builder:
                results.append(result_builder(item, responses[idx]))
            else:
                results.append(
                    {
                        "NL": item.get("NL"),
                        "FOL_PRED": responses[idx],
                        "FOL_GROUND-TRUTH": item.get("FOL"),
                    }
                )

        logger.info("Processed %d/%d examples", min(i + batch_size, len(data)), len(data))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("Inference complete. Results saved to %s", output_file)
    return results
