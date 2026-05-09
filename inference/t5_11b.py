

import json
import logging
import argparse
from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm

from utils.datasets.load_dataset import load_json_data
from utils.inference.encoder_decoder import restore_fol_symbols, prepare_t5_data, tokenize_t5_dataset
from utils.prompts import load_system_prompt_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_batch(batch, pad_token_id):
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]

    max_len = max(ids.size(0) for ids in input_ids_list)

    padded_input_ids = torch.full(
        (len(batch), max_len), pad_token_id, dtype=input_ids_list[0].dtype
    )
    padded_attention_mask = torch.zeros(
        (len(batch), max_len), dtype=attention_mask_list[0].dtype
    )

    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        padded_input_ids[i, : ids.size(0)] = ids
        padded_attention_mask[i, : mask.size(0)] = mask

    return padded_input_ids, padded_attention_mask


def main():
    """Run T5-11B inference on the goldstandard dataset and store predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--adapter_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--pred_key", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prefix_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    args = parser.parse_args()
    
    T5_SYSTEM_PROMPT = load_system_prompt_from_file(args.prefix_path)

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path, device_map="auto", is_trainable=False)
    model = model.merge_and_unload()
    model.eval()

    input_device = next(model.parameters()).device

    test_data = load_json_data(args.test_data_path)
    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.map(
        lambda x: prepare_t5_data(x, T5_SYSTEM_PROMPT), batched=False
    )
    test_tokenized = tokenize_t5_dataset(test_dataset, tokenizer)
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    original_data = load_json_data(args.output_file)
    pad_token_id = tokenizer.pad_token_id

    all_predictions = []
    num_entries = len(test_tokenized)

    for start_idx in tqdm(range(0, num_entries, args.batch_size), desc="Processing batches"):
        end_idx = min(start_idx + args.batch_size, num_entries)
        batch = [test_tokenized[i] for i in range(start_idx, end_idx)]

        input_ids, attention_mask = collate_batch(batch, pad_token_id)
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                min_length=1,
                num_beams=5,
                length_penalty=2.0,
                early_stopping=False,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(restore_fol_symbols(p) for p in decoded)

    assert len(all_predictions) == len(original_data), (
        f"Prediction count mismatch: {len(all_predictions)} vs {len(original_data)}"
    )

    for original_entry, prediction in zip(original_data, all_predictions, strict=True):
        original_entry[args.pred_key] = prediction

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(original_data, f, indent=4, ensure_ascii=False)

    logger.info("Inference complete. Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()