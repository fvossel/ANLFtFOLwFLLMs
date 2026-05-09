"""Inference utilities for encoder-decoder (seq2seq) models like T5."""

import json
import logging

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset

logger = logging.getLogger(__name__)

FOL_SYMBOL_TO_TOKEN = {
    "\u2200": "FORALL",
    "\u2203": "EXISTS",
    "\u2227": "AND",
    "\u2295": "XOR",
    "\u2228": "OR",
    "\u2192": "IMPLIES",
    "\u2194": "IFF",
    "\u00ac": "NOT",
    "+": "PLUS",
    "-": "MINUS",
    "*": "MULT",
    "/": "DIV",
    "<": "LESST",
    ">": "GREATT",
    "=": "EQUAL",
    "\u2264": "LESSE",
    "\u2265": "GREATE",
    "\u2260": "NEQ",
}

FOL_TOKEN_TO_SYMBOL = {v: k for k, v in FOL_SYMBOL_TO_TOKEN.items()}


def replace_fol_symbols(text: str) -> str:
    """Replace FOL symbols with text tokens for T5 processing."""
    for symbol, token in FOL_SYMBOL_TO_TOKEN.items():
        text = text.replace(symbol, token)
    return text


def restore_fol_symbols(text: str) -> str:
    """Restore FOL text tokens back to their symbol representations."""
    for token, symbol in FOL_TOKEN_TO_SYMBOL.items():
        text = text.replace(token, symbol)
    return text


def prepare_t5_data(example: dict, system_prompt: str) -> dict:
    """Prepare a single example for T5 inference by replacing FOL symbols and adding the system prompt."""
    fol = example["FOL"]
    for symbol, token in FOL_SYMBOL_TO_TOKEN.items():
        fol = fol.replace(symbol, token)
    return {
        "NL": system_prompt + example["NL"],
        "FOL": fol,
    }


def tokenize_t5_dataset(dataset: Dataset, tokenizer: T5Tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize a dataset for T5 models."""
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["NL"], padding="max_length", truncation=True, max_length=max_length
        )
        labels = tokenizer(
            examples["FOL"], padding="max_length", truncation=True, max_length=max_length
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def run_t5_inference(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    tokenized_dataset: Dataset,
    output_file: str,
    restore_symbols: bool = True,
) -> list[dict]:
    """Run inference with a T5 model on a tokenized dataset.

    Args:
        model: The loaded T5 model.
        tokenizer: The loaded T5 tokenizer.
        tokenized_dataset: Tokenized dataset with input_ids, attention_mask, labels.
        output_file: Path to save results.
        restore_symbols: Whether to restore FOL symbols in predictions.

    Returns:
        List of result dictionaries.
    """
    device = next(model.parameters()).device
    results = []
    total = len(tokenized_dataset)

    for i in range(total):
        input_ids = tokenized_dataset[i]["input_ids"].unsqueeze(0).to(device)
        labels = tokenized_dataset[i]["labels"].unsqueeze(0).to(device)

        sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        nl_sentence = " ".join(
            token.split("_")[0]
            for token in sentence.replace(
                "translate English natural language statements into first-order logic (FOL): ",
                "",
            ).split()
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                min_length=1,
                num_beams=5,
                length_penalty=2.0,
                early_stopping=False,
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if restore_symbols:
            prediction = restore_fol_symbols(prediction)

        ground_truth = tokenizer.decode(labels[0], skip_special_tokens=True)
        if restore_symbols:
            ground_truth = restore_fol_symbols(ground_truth)

        results.append(
            {
                "NL": nl_sentence,
                "FOL_PRED": prediction,
                "FOL_GROUND-TRUTH": ground_truth,
            }
        )

        if (i + 1) % 50 == 0 or i == total - 1:
            logger.info("Processed %d/%d examples", i + 1, total)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("Inference complete. Results saved to %s", output_file)
    return results
