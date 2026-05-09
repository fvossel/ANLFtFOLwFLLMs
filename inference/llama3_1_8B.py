

import logging
import argparse
from dotenv import load_dotenv
load_dotenv()

from utils.datasets.load_dataset import load_json_data
from utils.inference.decoder_only import load_decoder_only_for_inference, run_batch_inference
from utils.prompts import load_system_prompt_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run inference on the goldstandard dataset and store predictions under the model key."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--adapter_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pred_key", type=str)
    parser.add_argument("--systemprompt_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    systemprompt = load_system_prompt_from_file(args.systemprompt_path)

    model, tokenizer = load_decoder_only_for_inference(
        args.model_name, adapter_path=args.adapter_path, merge_adapter=True
    )

    data = load_json_data(args.file_path)

    def formatting_func(example):
        """Format an example as a chat prompt for inference."""
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": example["NL"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def result_builder(item, response):
        """Store the prediction under the model-specific key in the original entry."""
        item[args.pred_key] = response.split("𝜙=")[-1]
        return item

    run_batch_inference(
        model=model,
        tokenizer=tokenizer,
        data=data,
        formatting_func=formatting_func,
        output_file=args.output_path,
        result_builder=result_builder,
        strip_input=True,
    )


if __name__ == "__main__":
    main()
