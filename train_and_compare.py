"""
train_and_compare.py
--------------------
Runs BERT MLM training under two (or more) masking strategies and
produces a side-by-side comparison of eval perplexity.

Usage
-----
# Default: baseline + span-masking, 1 epoch each
python train_and_compare.py

# Span-length ablation (experiments 3 and 4 from the paper)
python train_and_compare.py --mode span_ablation

# Masking-rate ablation
python train_and_compare.py --mode rate_ablation

All results are written to results/comparison_results.json so you can
load them into the notebook / widget afterwards.
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from span_masking_collator import SpanMaskingDataCollator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
DATASET_FRACTION = "train[:2000]"  # Reduced from 5000 for faster training
OUTPUT_ROOT = "./results"


@dataclass
class ExperimentConfig:
    name: str
    masking_type: str          # "baseline" | "span"
    mlm_probability: float = 0.15
    mean_span_length: float = 3.0
    max_span_length: int = 10
    num_epochs: int = 1
    batch_size: int = 4


# ---------------------------------------------------------------------------
# Experiment suites
# ---------------------------------------------------------------------------

BASELINE_VS_SPAN: List[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline_15pct",
        masking_type="baseline",
        mlm_probability=0.15,
    ),
    ExperimentConfig(
        name="span_15pct_mean3",
        masking_type="span",
        mlm_probability=0.15,
        mean_span_length=3.0,
    ),
]

SPAN_LENGTH_ABLATION: List[ExperimentConfig] = [
    ExperimentConfig(
        name="span_mean1",
        masking_type="span",
        mlm_probability=0.15,
        mean_span_length=1.0,   # degenerates toward token-level masking
    ),
    ExperimentConfig(
        name="span_mean3",
        masking_type="span",
        mlm_probability=0.15,
        mean_span_length=3.0,   # SpanBERT default
    ),
    ExperimentConfig(
        name="span_mean5",
        masking_type="span",
        mlm_probability=0.15,
        mean_span_length=5.0,
    ),
    ExperimentConfig(
        name="span_mean10",
        masking_type="span",
        mlm_probability=0.15,
        mean_span_length=10.0,
    ),
]

MASKING_RATE_ABLATION: List[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline_10pct",
        masking_type="baseline",
        mlm_probability=0.10,
    ),
    ExperimentConfig(
        name="baseline_15pct",
        masking_type="baseline",
        mlm_probability=0.15,
    ),
    ExperimentConfig(
        name="baseline_20pct",
        masking_type="baseline",
        mlm_probability=0.20,
    ),
    ExperimentConfig(
        name="span_10pct",
        masking_type="span",
        mlm_probability=0.10,
        mean_span_length=3.0,
    ),
    ExperimentConfig(
        name="span_15pct",
        masking_type="span",
        mlm_probability=0.15,
        mean_span_length=3.0,
    ),
    ExperimentConfig(
        name="span_20pct",
        masking_type="span",
        mlm_probability=0.20,
        mean_span_length=3.0,
    ),
]

EXPERIMENT_SUITES = {
    "baseline_vs_span": BASELINE_VS_SPAN,
    "span_ablation": SPAN_LENGTH_ABLATION,
    "rate_ablation": MASKING_RATE_ABLATION,
}


# ---------------------------------------------------------------------------
# Dataset helpers (shared across runs to save time)
# ---------------------------------------------------------------------------

def load_and_prepare_datasets(tokenizer):
    """Load Wikipedia subset, tokenize, chunk, and split once."""
    print("[data] Loading dataset …")
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split=DATASET_FRACTION
    )
    dataset = dataset.filter(
        lambda x: x["text"] is not None and len(x["text"].strip()) > 0
    )

    print("[data] Tokenizing …")
    tokenized = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )

    print("[data] Chunking into fixed-length sequences …")
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concatenated["input_ids"]) // MAX_LENGTH) * MAX_LENGTH
        return {
            k: [t[i: i + MAX_LENGTH] for i in range(0, total, MAX_LENGTH)]
            for k, t in concatenated.items()
        }

    lm_dataset = tokenized.map(group_texts, batched=True)
    split = lm_dataset.train_test_split(test_size=0.02, seed=42)
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: ExperimentConfig,
    train_dataset,
    eval_dataset,
    tokenizer,
) -> dict:
    """Train one model configuration and return the eval metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {cfg.name}")
    print(f"  Type      : {cfg.masking_type}")
    print(f"  mask prob : {cfg.mlm_probability}")
    if cfg.masking_type == "span":
        print(f"  mean span : {cfg.mean_span_length}")
    print(f"{'='*60}\n")

    # Fresh model for each experiment
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # Build the appropriate collator
    if cfg.masking_type == "baseline":
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=cfg.mlm_probability,
        )
    else:
        collator = SpanMaskingDataCollator(
            tokenizer=tokenizer,
            mlm_probability=cfg.mlm_probability,
            mean_span_length=cfg.mean_span_length,
            max_span_length=cfg.max_span_length,
        )

    output_dir = os.path.join(OUTPUT_ROOT, cfg.name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        logging_steps=100,  # Reduced checkpoint logging for speed
        save_steps=1000,    # Reduced checkpoint frequency
        save_total_limit=1,
        report_to="none",
        # Evaluate at the end of training
        eval_strategy="epoch",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # Add perplexity (not computed by default)
    loss = eval_metrics.get("eval_loss", float("nan"))
    eval_metrics["eval_perplexity"] = math.exp(loss) if not math.isnan(loss) else float("nan")

    result = {
        "config": asdict(cfg),
        "metrics": eval_metrics,
    }
    print(f"\n[result] {cfg.name}: loss={loss:.4f}  ppl={eval_metrics['eval_perplexity']:.2f}")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BERT MLM masking strategy comparison")
    parser.add_argument(
        "--mode",
        choices=list(EXPERIMENT_SUITES.keys()),
        default="baseline_vs_span",
        help="Which experiment suite to run.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_ROOT, "comparison_results.json"),
        help="Path to write JSON results.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    configs = EXPERIMENT_SUITES[args.mode]
    print(f"\nRunning suite '{args.mode}' ({len(configs)} experiments)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, eval_dataset = load_and_prepare_datasets(tokenizer)

    all_results = []
    for cfg in configs:
        result = run_experiment(cfg, train_dataset, eval_dataset, tokenizer)
        all_results.append(result)

        # Save incrementally so partial results aren't lost
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    # ---- Summary table -----------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {args.mode}")
    print(f"{'='*60}")
    print(f"  {'Experiment':<30} {'Loss':>8} {'Perplexity':>12}")
    print(f"  {'-'*52}")
    for r in all_results:
        name = r["config"]["name"]
        loss = r["metrics"].get("eval_loss", float("nan"))
        ppl  = r["metrics"].get("eval_perplexity", float("nan"))
        print(f"  {name:<30} {loss:>8.4f} {ppl:>12.2f}")
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
