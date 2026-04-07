from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True,
    )

def group_texts(examples):
    # Concatenate all texts, then split into equal chunks.
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])

    # Drop remainder so every chunk is same length
    total_length = (total_length // MAX_LENGTH) * MAX_LENGTH

    result = {
        k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
        for k, t in concatenated.items()
    }
    return result

def main():
    # 1) Load a small Wikipedia subset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]")

    # Keep only rows that actually have text
    dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)

    # 2) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # 3) Tokenize raw text
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # 4) Group tokens into fixed-size chunks
    lm_dataset = tokenized.map(group_texts, batched=True)

    # Train/validation split for a quick test
    split = lm_dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # 5) Standard baseline masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 6) Training config
    training_args = TrainingArguments(
        output_dir="./bert-baseline-mlm",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

if __name__ == "__main__":
    main()