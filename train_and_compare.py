"""
train_and_compare.py  (v2 — adds linguistic span masking)
----------------------------------------------------------
Runs BERT MLM training under three masking strategies:
  1. Baseline     — token-level, random 15% mask
  2. Geometric    — SpanBERT-style geometric span masking
  3. Linguistic   — NE > NP > VP span masking (this work)

Usage
-----
# Head-to-head: baseline vs geometric vs linguistic
python train_and_compare.py --mode all_three

# Linguistic ratio ablation (how much of budget is linguistic vs geometric)
python train_and_compare.py --mode ratio_ablation

# Original span-length and masking-rate ablations (unchanged)
python train_and_compare.py --mode span_ablation
python train_and_compare.py --mode rate_ablation

All results are written to results/comparison_results.json incrementally.
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
from linguistic_span_masking_collator import (
    LinguisticSpanMaskingDataCollator,
    tokenize_with_offsets,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
DATASET_FRACTION = "train[:2000]"
OUTPUT_ROOT = "./results"
SPACY_MODEL = "en_core_web_sm"


@dataclass
class ExperimentConfig:
    name: str
    masking_type: str           # "baseline" | "span" | "linguistic"
    mlm_probability: float = 0.15
    # Geometric span params
    mean_span_length: float = 3.0
    max_span_length: int = 10
    # Linguistic params
    linguistic_ratio: float = 1.0   # 1.0 = all ling, 0.0 = pure geometric fallback
    num_epochs: int = 1
    batch_size: int = 4


# ---------------------------------------------------------------------------
# Experiment suites
# ---------------------------------------------------------------------------

ALL_THREE: List[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline_15pct",
        masking_type="baseline",
        mlm_probability=0.15,
    ),
    ExperimentConfig(
        name="span_mean3_15pct",
        masking_type="span",
        mlm_probability=0.15,
        mean_span_length=3.0,
    ),
    ExperimentConfig(
        name="linguistic_NE_NP_VP",
        masking_type="linguistic",
        mlm_probability=0.15,
        linguistic_ratio=1.0,
    ),
]

# How much of the masking budget should come from linguistic spans?
RATIO_ABLATION: List[ExperimentConfig] = [
    ExperimentConfig(
        name="ling_ratio_0pct",
        masking_type="linguistic",
        mlm_probability=0.15,
        linguistic_ratio=0.0,   # pure geometric fallback
    ),
    ExperimentConfig(
        name="ling_ratio_25pct",
        masking_type="linguistic",
        mlm_probability=0.15,
        linguistic_ratio=0.25,
    ),
    ExperimentConfig(
        name="ling_ratio_50pct",
        masking_type="linguistic",
        mlm_probability=0.15,
        linguistic_ratio=0.50,
    ),
    ExperimentConfig(
        name="ling_ratio_75pct",
        masking_type="linguistic",
        mlm_probability=0.15,
        linguistic_ratio=0.75,
    ),
    ExperimentConfig(
        name="ling_ratio_100pct",
        masking_type="linguistic",
        mlm_probability=0.15,
        linguistic_ratio=1.0,
    ),
]

SPAN_LENGTH_ABLATION: List[ExperimentConfig] = [
    ExperimentConfig(name="span_mean1",  masking_type="span", mlm_probability=0.15, mean_span_length=1.0),
    ExperimentConfig(name="span_mean3",  masking_type="span", mlm_probability=0.15, mean_span_length=3.0),
    ExperimentConfig(name="span_mean5",  masking_type="span", mlm_probability=0.15, mean_span_length=5.0),
    ExperimentConfig(name="span_mean10", masking_type="span", mlm_probability=0.15, mean_span_length=10.0),
]

MASKING_RATE_ABLATION: List[ExperimentConfig] = [
    ExperimentConfig(name="baseline_10pct",  masking_type="baseline",    mlm_probability=0.10),
    ExperimentConfig(name="baseline_15pct",  masking_type="baseline",    mlm_probability=0.15),
    ExperimentConfig(name="baseline_20pct",  masking_type="baseline",    mlm_probability=0.20),
    ExperimentConfig(name="span_10pct",      masking_type="span",        mlm_probability=0.10, mean_span_length=3.0),
    ExperimentConfig(name="span_15pct",      masking_type="span",        mlm_probability=0.15, mean_span_length=3.0),
    ExperimentConfig(name="span_20pct",      masking_type="span",        mlm_probability=0.20, mean_span_length=3.0),
    ExperimentConfig(name="linguistic_10pct",masking_type="linguistic",  mlm_probability=0.10, linguistic_ratio=1.0),
    ExperimentConfig(name="linguistic_15pct",masking_type="linguistic",  mlm_probability=0.15, linguistic_ratio=1.0),
    ExperimentConfig(name="linguistic_20pct",masking_type="linguistic",  mlm_probability=0.20, linguistic_ratio=1.0),
]

EXPERIMENT_SUITES = {
    "all_three":      ALL_THREE,
    "ratio_ablation": RATIO_ABLATION,
    "span_ablation":  SPAN_LENGTH_ABLATION,
    "rate_ablation":  MASKING_RATE_ABLATION,
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_and_prepare_datasets(tokenizer, needs_offsets: bool = False):
    """
    Load, filter, tokenize, chunk, and split the Wikipedia dataset.

    If needs_offsets=True, the tokenizer call also stores offset_mapping
    and original_text so the linguistic collator can run spaCy at collation
    time. This adds a small memory overhead but avoids running spaCy over
    the entire dataset up front.
    """
    print("[data] Loading dataset …")
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split=DATASET_FRACTION
    )
    dataset = dataset.filter(
        lambda x: x["text"] is not None and len(x["text"].strip()) > 0
    )

    print("[data] Tokenizing …")
    if needs_offsets:
        # Use offset-aware tokenizer for linguistic collator
        tokenized = dataset.map(
            lambda examples: tokenize_with_offsets(examples, tokenizer, MAX_LENGTH),
            batched=True,
            remove_columns=[c for c in dataset.column_names if c != "text"],
        )
        # remove_columns above keeps "text"; rename it to "original_text" for clarity
        # tokenize_with_offsets already adds original_text, so we can drop "text"
        tokenized = tokenized.remove_columns(["text"])
    else:
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

    print("[data] Chunking …")
    if needs_offsets:
        # We cannot simply concatenate offset_mapping lists, so we skip
        # the grouping step for the linguistic collator — each article
        # is processed as a single (possibly truncated) sequence.
        # This trades some training efficiency for correct spaCy alignment.
        lm_dataset = tokenized
    else:
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
    tokenizer,
    nlp=None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Experiment : {cfg.name}")
    print(f"  Type       : {cfg.masking_type}")
    print(f"  mask prob  : {cfg.mlm_probability}")
    if cfg.masking_type == "span":
        print(f"  mean span  : {cfg.mean_span_length}")
    if cfg.masking_type == "linguistic":
        print(f"  ling ratio : {cfg.linguistic_ratio}")
    print(f"{'='*60}\n")

    needs_offsets = cfg.masking_type == "linguistic"
    train_dataset, eval_dataset = load_and_prepare_datasets(tokenizer, needs_offsets)

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    if cfg.masking_type == "baseline":
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=cfg.mlm_probability,
        )
    elif cfg.masking_type == "span":
        collator = SpanMaskingDataCollator(
            tokenizer=tokenizer,
            mlm_probability=cfg.mlm_probability,
            mean_span_length=cfg.mean_span_length,
            max_span_length=cfg.max_span_length,
        )
    else:  # linguistic
        if nlp is None:
            raise ValueError("nlp (spaCy model) required for linguistic masking.")
        collator = LinguisticSpanMaskingDataCollator(
            tokenizer=tokenizer,
            nlp=nlp,
            mlm_probability=cfg.mlm_probability,
            linguistic_ratio=cfg.linguistic_ratio,
            mean_span_length=cfg.mean_span_length,
            max_span_length=cfg.max_span_length,
        )

    output_dir = os.path.join(OUTPUT_ROOT, cfg.name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
        report_to="none",
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

    loss = eval_metrics.get("eval_loss", float("nan"))
    eval_metrics["eval_perplexity"] = math.exp(loss) if not math.isnan(loss) else float("nan")

    result = {"config": asdict(cfg), "metrics": eval_metrics}
    print(f"\n[result] {cfg.name}: loss={loss:.4f}  ppl={eval_metrics['eval_perplexity']:.2f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(EXPERIMENT_SUITES.keys()),
        default="all_three",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_ROOT, "comparison_results.json"),
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    configs = EXPERIMENT_SUITES[args.mode]
    print(f"\nRunning suite '{args.mode}' ({len(configs)} experiments)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load spaCy only if at least one experiment needs it
    nlp = None
    if any(c.masking_type == "linguistic" for c in configs):
        try:
            import spacy
            print(f"[spacy] Loading {SPACY_MODEL} …")
            nlp = spacy.load(SPACY_MODEL)
            print("[spacy] Loaded.\n")
        except OSError:
            print(
                f"[spacy] Model '{SPACY_MODEL}' not found.\n"
                f"        Run: python -m spacy download {SPACY_MODEL}\n"
            )
            raise

    all_results = []
    for cfg in configs:
        result = run_experiment(cfg, tokenizer, nlp)
        all_results.append(result)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
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