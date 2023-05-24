import multiprocessing
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

class CFG:
    SEQ_LEN: int = 1024
    NUM_CPU: int = multiprocessing.cpu_count()
    TOKENIZER: str = "gpt2"

def build_dataset():
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

    processed_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=CFG.NUM_CPU,
    )

    # Save the preprocessed dataset
    processed_dataset.save_to_disk("dataset")

if __name__ == '__main__':
    build_dataset()