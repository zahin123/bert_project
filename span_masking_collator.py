"""
SpanMaskingDataCollator
-----------------------
Replaces HuggingFace's token-level MLM collator with span-level masking.

Instead of masking random *individual* tokens at a fixed probability,
we sample contiguous *spans* of tokens using a geometric distribution
for the span length (as in SpanBERT, Joshi et al. 2020).

Key hyperparameters
~~~~~~~~~~~~~~~~~~~
mlm_probability   : target fraction of tokens to mask in total (default 0.15)
mean_span_length  : geometric distribution mean (default 3.0, SpanBERT default)
max_span_length   : hard cap on any single span (default 10)

Reference: SpanBERT — https://arxiv.org/abs/1907.10529
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class SpanMaskingDataCollator:
    """
    Data collator that performs span-level masked language modeling.

    Works as a drop-in replacement for DataCollatorForLanguageModeling
    when mlm=True.

    Parameters
    ----------
    tokenizer         : HuggingFace tokenizer (must have mask_token).
    mlm_probability   : Target fraction of input tokens to mask (≈ 0.15).
    mean_span_length  : Mean of the geometric distribution used to sample
                        span lengths.  SpanBERT uses 3.0.
    max_span_length   : Hard upper bound on a single span (avoids masking
                        the whole sequence on a rare long draw).
    mask_probability  : Within a selected span, probability of replacing
                        with [MASK].  The remainder splits evenly between
                        keeping original and replacing with a random token
                        (80 / 10 / 10 split, same as BERT baseline).
    pad_to_multiple_of: Optional — pads batch length to this multiple.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    mean_span_length: float = 3.0
    max_span_length: int = 10
    mask_probability: float = 0.80          # prob of using [MASK] vs orig/random
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token — "
                "span masking requires a tokenizer with a [MASK] token."
            )
        # Geometric distribution parameter: p = 1 / mean_span_length
        self._geo_p = 1.0 / self.mean_span_length

    # ------------------------------------------------------------------
    # Public interface (matches DataCollatorForLanguageModeling)
    # ------------------------------------------------------------------

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a list of tokenized samples into a batch with span masks.

        Each feature dict should have at minimum:
          - "input_ids"            : List[int]
          - "attention_mask"       : List[int]  (optional but typical)
          - "special_tokens_mask"  : List[int]  (1 = special, 0 = normal)
        """
        # ---- 1. Pad / stack into tensors --------------------------------
        batch = self._pad_batch(features)

        # ---- 2. Clone input_ids as labels; mask positions set to -100 ---
        labels = batch["input_ids"].clone()

        # ---- 3. Apply span masking row-by-row ---------------------------
        for i in range(labels.size(0)):
            batch["input_ids"][i], labels[i] = self._mask_single_sequence(
                batch["input_ids"][i],
                special_tokens_mask=batch.get("special_tokens_mask", None),
                row_index=i,
            )

        batch["labels"] = labels

        # Drop the special_tokens_mask — the model doesn't need it
        batch.pop("special_tokens_mask", None)
        return batch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mask_single_sequence(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor],
        row_index: int,
    ):
        """
        Apply span masking to one sequence.

        Returns
        -------
        masked_ids : torch.Tensor  — input_ids with selected tokens replaced
        labels     : torch.Tensor  — original ids at masked positions, -100 elsewhere
        """
        seq_len = input_ids.size(0)
        labels = input_ids.clone()

        # Boolean mask: True = this position can be masked
        if special_tokens_mask is not None:
            can_mask = special_tokens_mask[row_index].eq(0)
        else:
            # Derive from tokenizer if not provided
            sp_mask = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            can_mask = torch.tensor(sp_mask, dtype=torch.bool).logical_not()

        # Track which positions have been masked
        is_masked = torch.zeros(seq_len, dtype=torch.bool)

        # Budget: how many tokens should we mask in total?
        n_maskable = int(can_mask.sum().item())
        target_masked = max(1, round(n_maskable * self.mlm_probability))

        # ---- Span sampling loop ----------------------------------------
        # Collect candidate start positions (non-special, not yet masked)
        attempts = 0
        while is_masked.sum().item() < target_masked and attempts < seq_len * 4:
            attempts += 1

            # Sample span length from a geometric distribution
            span_len = self._sample_span_length()

            # Choose a random start position
            start = random.randint(0, seq_len - 1)
            if not can_mask[start]:
                continue

            # Extend span rightward while positions are valid
            end = start
            for j in range(start, min(start + span_len, seq_len)):
                if can_mask[j] and not is_masked[j]:
                    end = j
                else:
                    break

            span = slice(start, end + 1)
            is_masked[span] = True

        # ---- Apply BERT 80/10/10 replacement rule ----------------------
        masked_ids = input_ids.clone()
        masked_positions = is_masked.nonzero(as_tuple=True)[0]

        for pos in masked_positions:
            r = random.random()
            if r < self.mask_probability:
                # Replace with [MASK]
                masked_ids[pos] = self.tokenizer.mask_token_id
            elif r < self.mask_probability + (1 - self.mask_probability) / 2:
                # Keep original (no-op; already cloned)
                pass
            else:
                # Replace with a random vocabulary token
                masked_ids[pos] = random.randint(0, self.tokenizer.vocab_size - 1)

        # Positions not masked get label = -100 (ignored in cross-entropy)
        labels[~is_masked] = -100

        return masked_ids, labels

    def _sample_span_length(self) -> int:
        """
        Draw a span length from a geometric distribution, capped at
        max_span_length.
        """
        # Clamp p away from 1.0 to avoid log(0) when mean_span_length = 1
        p = min(self._geo_p, 0.9999)
        length = 0
        while length < 1:
            r = random.random()
            if r < 1e-12:
                r = 1e-12
            length = int(math.ceil(math.log(r) / math.log(1 - p)))
        return min(length, self.max_span_length)

    def _pad_batch(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Pad a list of feature dicts to the same length and stack into tensors.
        Mirrors what DataCollatorWithPadding does internally.
        """
        # Collect all keys (except special_tokens_mask which may be absent)
        keys = list(features[0].keys())

        # Find max length in this batch
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            max_len = (
                math.ceil(max_len / self.pad_to_multiple_of)
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id or 0
        batch: Dict[str, torch.Tensor] = {}

        for key in keys:
            pad_value = 0 if key != "input_ids" else pad_id
            # special_tokens_mask uses 1 for padding positions
            if key == "special_tokens_mask":
                pad_value = 1

            padded = []
            for f in features:
                seq = list(f[key])
                seq += [pad_value] * (max_len - len(seq))
                padded.append(seq)

            batch[key] = torch.tensor(padded, dtype=torch.long)

        return batch
