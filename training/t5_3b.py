from dotenv import load_dotenv
load_dotenv()
import argparse
from transformers import T5Tokenizer, TrainingArguments, Trainer, T5ForConditionalGeneration
import torch
from utils.datasets.load_dataset import load_dataset_for_encoder_decoder_model
from transformers import EarlyStoppingCallback, TrainerCallback

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google-t5/t5-3b")
parser.add_argument("--system_prompt", required=True)
parser.add_argument("--train_path", required=True)
parser.add_argument("--eval_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--final_model", required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(args.model)

with open(args.system_prompt, "r", encoding="utf-8") as f:
    system_prompt = f.read()

train_dataset_tokenized, val_dataset_tokenized = load_dataset_for_encoder_decoder_model(
    system_prompt=system_prompt,
    tokenizer=tokenizer,
    train_path=args.train_path,
    eval_path=args.eval_path,
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=12,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=12,
    learning_rate=1e-4,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    warmup_steps=500,
    gradient_accumulation_steps=1,
    disable_tqdm=False
)

model = T5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).to(device)


class BestModelTracker(TrainerCallback):
    def __init__(self):
        self.best_epoch = None
        self.best_eval_loss = float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_loss", float('inf')) < self.best_eval_loss:
            self.best_eval_loss = metrics["eval_loss"]
            self.best_epoch = state.epoch

best_model_tracker = BestModelTracker()

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=4),
        best_model_tracker
    ],
)

# Start training
trainer.train()

# Save the trained model and tokenizer
trainer.save_model(args.final_model)
tokenizer.save_pretrained(args.final_model)

print("Training complete!")
print(f"Best model saved at epoch {best_model_tracker.best_epoch:.2f} with eval loss {best_model_tracker.best_eval_loss:.4f}")