import torch
import multiprocessing
from itertools import chain
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from Sophia.decoupled_sophia.decoupled_sophia import DecoupledSophia, HutchinsonEstimator
from transformers import AutoTokenizer

# Load and preprocess the OpenWebText dataset
class CFG:
    SEQ_LEN: int = 1024
    NUM_CPU: int = multiprocessing.cpu_count()
    TOKENIZER: str = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(CFG.TOKENIZER)
dataset = load_dataset("openwebtext")

def tokenize_function(example):
    return tokenizer(example["text"] + tokenizer.eos_token)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=CFG.NUM_CPU,
    remove_columns=["text"],
)

block_size = CFG.SEQ_LEN

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

train_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=CFG.NUM_CPU,
)

# Initialize the GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2", n_ctx=1024)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# Choose a Hessian estimator
hessian_estimator = HutchinsonEstimator()

# Initialize the DecoupledSophia optimizer
optimizer = DecoupledSophia(model.parameters(), hessian_estimator, lr=1e-3)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=480,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    gradient_accumulation_steps=1,
    gradient_clipping=1.0,
    learning_rate_scheduler_type="cosine",
    warmup_steps=2000,
    report_to="none",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=train_dataset,
    optimizers=(optimizer, None),
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")