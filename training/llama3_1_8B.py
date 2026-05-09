from dotenv import load_dotenv
load_dotenv()
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from transformers import EarlyStoppingCallback, TrainerCallback
from utils.datasets.load_dataset import load_dataset_for_decoder_only_model
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--system_prompt", required=True)
parser.add_argument("--lora_config", required=True)
parser.add_argument("--train_path", required=True)
parser.add_argument("--eval_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--final_model", required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    use_cache=False,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

with open(args.system_prompt, "r", encoding="utf-8") as f:
    system_prompt = f.read()

train_dataset, val_dataset = load_dataset_for_decoder_only_model(
    system_prompt, args.train_path, args.eval_path
)

with open(args.lora_config, "r", encoding="utf-8") as f:
    lora_config_dict = json.load(f)

lora_config = LoraConfig(**lora_config_dict)


#  Configure the Trainer with Early Stopping and Best Model Tracking

training_args = TrainingArguments(
    output_dir=args.output_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=12,
    weight_decay=0.01,
    bf16=True, 
    save_total_limit=12,
    logging_steps=100,
    report_to="none",
)

class BestModelTracker(TrainerCallback):
    def __init__(self):
        self.best_epoch = None
        self.best_eval_loss = float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_loss", float('inf')) < self.best_eval_loss:
            self.best_eval_loss = metrics["eval_loss"]
            self.best_epoch = state.epoch

best_model_tracker = BestModelTracker()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=4),
        best_model_tracker
    ],
)

#  Start training
trainer.train()

#  Save the best model
trainer.save_model(args.final_model)

print("Training complete!")
print(f"Best model saved at epoch {best_model_tracker.best_epoch:.2f} with eval loss {best_model_tracker.best_eval_loss:.4f}")
