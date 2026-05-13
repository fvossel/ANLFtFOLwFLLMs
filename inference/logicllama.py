"""Zero-shot FOL translation inference using LogicLLaMA-7b.

LogicLLaMA uses a PEFT adapter on top of the huggyllama/llama-7b base model with custom
tokenizer special tokens and a fixed GenerationConfig (temperature, top_p, top_k). Because
these requirements differ from the shared decoder-only utility, model loading and generation
are handled inline rather than delegated to utils.inference.decoder_only.
"""

import json
import logging
import argparse
from dotenv import load_dotenv
load_dotenv()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils.datasets.load_dataset import load_json_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOGICLLAMA_GENERATION_CONFIG = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=512,
)


def load_logicllama_for_inference(model_name: str, peft_path: str) -> tuple:
    """Load the LogicLLaMA model and tokenizer for inference.

    Registers the required special tokens and applies the PEFT adapter without
    merging, since LogicLLaMA inference uses the adapter in delta mode.

    Args:
        model_name: HuggingFace identifier for the base LLaMA model.
        peft_path: HuggingFace identifier or local path to the PEFT adapter.

    Returns:
        Tuple of (model, tokenizer) ready for generation on CUDA.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, legacy=False)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
    })
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_cache=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        peft_path,
        torch_dtype=torch.float16,
        use_cache=True,
        trust_remote_code=True,
    )
    model.to("cuda")
    logger.info("Loaded LogicLLaMA from %s with adapter %s", model_name, peft_path)
    return model, tokenizer


def main():
    """Run inference on the goldstandard dataset and store predictions under the model key."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--adapter_path", type=str)
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--pred_key", type=str)
    parser.add_argument("--systemprompt_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    with open(args.systemprompt_path, "r", encoding="utf-8") as f:
        systemprompt = f.read()
    model, tokenizer = load_logicllama_for_inference(args.model_name, args.adapter_path)
    data = load_json_data(args.file_path)

    def formatting_func(example):
        """Format an example as a chat prompt for LogicLLaMA inference."""
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": example["NL"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    batch_size = 24
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
                generation_config=LOGICLLAMA_GENERATION_CONFIG,
            )

        responses = [
            tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in outputs
        ]

        for idx, item in enumerate(batch_data):
            item[args.pred_key] = responses[idx]
            results.append(item)

        logger.info("Processed %d/%d examples", min(i + batch_size, len(data)), len(data))

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("Inference complete. Results saved to %s", args.output_path)


if __name__ == "__main__":
    main()
