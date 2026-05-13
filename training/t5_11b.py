"""LoRA fine-tuning of Flan-T5-XXL (11B) for NL→FOL translation."""
from dotenv import load_dotenv
load_dotenv()
import argparse
import os
from transformers import T5Tokenizer, TrainingArguments, T5ForConditionalGeneration, Trainer
import torch
import json
from peft import LoraConfig, get_peft_model, PeftModel
from utils.datasets.load_dataset import load_dataset_for_encoder_decoder_model
from utils.inference.encoder_decoder import FOL_SYMBOL_TO_TOKEN
from transformers import EarlyStoppingCallback, TrainerCallback

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/flan-t5-xxl")
parser.add_argument("--system_prompt", required=True)
parser.add_argument("--lora_config", required=True)
parser.add_argument("--train_path", required=True)
parser.add_argument("--eval_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--final_model", required=True)
parser.add_argument("--adapter_path", default=None)
parser.add_argument("--token_extension", action="store_true")
args = parser.parse_args()

tokenizer = T5Tokenizer.from_pretrained(args.model)

model = T5ForConditionalGeneration.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
)

if args.token_extension:
    fol_special_tokens = list(FOL_SYMBOL_TO_TOKEN.values())
    tokenizer.add_tokens(fol_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(os.path.join(args.output_dir, "base_model_with_embeddings"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "base_model_with_embeddings"))

with open(args.system_prompt, "r", encoding="utf-8") as f:
    system_prompt = f.read()

train_dataset_tokenized, val_dataset_tokenized = load_dataset_for_encoder_decoder_model(
    system_prompt=system_prompt,
    tokenizer=tokenizer,
    train_path=args.train_path,
    eval_path=args.eval_path,
)

with open(args.lora_config, "r", encoding="utf-8") as f:
    lora_config_dict = json.load(f)

lora_config = LoraConfig(**lora_config_dict)
if args.adapter_path is not None:
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    model.train()
else:
    model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    save_total_limit=12,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=12,
    eval_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    warmup_steps=500,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    disable_tqdm=False,
)


class BestModelTracker(TrainerCallback):
    """Tracks the best checkpoint by eval loss across training epochs."""

    def __init__(self):
        self.best_epoch = None
        self.best_eval_loss = float('inf')

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_loss", float('inf')) < self.best_eval_loss:
            self.best_eval_loss = metrics["eval_loss"]
            self.best_epoch = state.epoch


best_model_tracker = BestModelTracker()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=4),
        best_model_tracker,
    ],
)

trainer.train(resume_from_checkpoint=True)
trainer.save_model(args.final_model)
tokenizer.save_pretrained(args.final_model)

print("Training complete!")
print(f"Best model saved at epoch {best_model_tracker.best_epoch:.2f} with eval loss {best_model_tracker.best_eval_loss:.4f}")
