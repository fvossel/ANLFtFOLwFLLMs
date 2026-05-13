"""Dataset loading and preparation utilities for FOL translation experiments."""

import json
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer

from utils.inference.encoder_decoder import FOL_SYMBOL_TO_TOKEN


def load_json_dataset(file_path: str) -> Dataset:
    """Load a JSON file and convert it into a HuggingFace Dataset with NL and FOL columns."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    features = Features({
        "NL": Value("string"),
        "FOL": Value("string"),
    })
    return Dataset.from_list(data, features=features)


def load_json_data(file_path: str) -> list[dict]:
    """Load a JSON file as a plain list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_prompt_generation_decoder_only(
    example: dict[str, str],
    tokenizer: AutoTokenizer,
    system_prompt: str,
) -> str:
    """Format an input example as a chat prompt for a decoder-only model."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["NL"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def format_messages_for_training(
    example: dict[str, str],
    system_prompt: str,
    fol_prefix: str = "\ud835\udf19=",
) -> dict:
    """Format an example as chat messages for SFT training of decoder-only models."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["NL"]},
            {"role": "assistant", "content": f"{fol_prefix}{example['FOL']}"},
        ]
    }


def load_dataset_for_decoder_only_model(
    system_prompt: str,
    train_path: str,
    eval_path: str,
    replace_fol_symbols: bool = False,
) -> tuple[Dataset, Dataset]:
    """Load and format train/eval datasets for decoder-only model fine-tuning."""
    train_dataset = load_json_dataset(train_path)
    val_dataset = load_json_dataset(eval_path)

    def formatting_func(example: dict[str, str]) -> dict:
        fol = example["FOL"]
        if replace_fol_symbols:
            for symbol, token in FOL_SYMBOL_TO_TOKEN.items():
                fol = fol.replace(symbol, token)
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["NL"]},
                {"role": "assistant", "content": fol},
            ]
        }

    train_dataset = train_dataset.map(
        formatting_func, remove_columns=["NL", "FOL"], batched=False
    )
    val_dataset = val_dataset.map(
        formatting_func, remove_columns=["NL", "FOL"], batched=False
    )
    return train_dataset, val_dataset


def prepare_encoder_decoder_data(example: dict[str, str], system_prompt: str) -> dict[str, str]:
    """Prepare an example for encoder-decoder models by replacing FOL symbols with text tokens."""
    fol = example["FOL"]
    for symbol, token in FOL_SYMBOL_TO_TOKEN.items():
        fol = fol.replace(symbol, token)
    return {
        "NL": system_prompt + example["NL"],
        "FOL": fol,
    }


def tokenize_encoder_decoder_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """Tokenize a dataset for encoder-decoder models."""
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


def load_dataset_for_encoder_decoder_model(
    system_prompt: str,
    tokenizer,
    train_path: str | None = None,
    eval_path: str | None = None,
    test_path: str | None = None,
) -> tuple[Dataset, Dataset] | Dataset:
    """Load and prepare datasets for encoder-decoder model training or testing.

    Pass train_path and eval_path for training, or test_path for evaluation.
    """
    if test_path is not None:
        test_dataset = load_json_dataset(test_path)
        test_dataset = test_dataset.map(
            lambda x: prepare_encoder_decoder_data(x, system_prompt), batched=False
        )
        return tokenize_encoder_decoder_dataset(test_dataset, tokenizer)

    if train_path is None or eval_path is None:
        raise ValueError("train_path and eval_path are required when test_path is not provided.")

    train_dataset = load_json_dataset(train_path)
    val_dataset = load_json_dataset(eval_path)

    train_dataset = train_dataset.map(
        lambda x: prepare_encoder_decoder_data(x, system_prompt), batched=False
    )
    val_dataset = val_dataset.map(
        lambda x: prepare_encoder_decoder_data(x, system_prompt), batched=False
    )

    return (
        tokenize_encoder_decoder_dataset(train_dataset, tokenizer),
        tokenize_encoder_decoder_dataset(val_dataset, tokenizer),
    )
