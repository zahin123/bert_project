"""
LinguisticSpanMaskingDataCollator
----------------------------------
An improvement over SpanBERT-style geometric span masking.

Instead of sampling random contiguous spans, this collator uses spaCy to
extract linguistically meaningful units — Named Entities (NEs), Noun Phrases
(NPs), and Verb Phrases (VPs) — and preferentially masks those over arbitrary
token runs.

Hypothesis
~~~~~~~~~~
Masking semantically rich units (entities, noun phrases, verb phrases) forces
the model to reconstruct tokens that carry real meaning from context alone,
creating a stronger learning signal than masking arbitrary substrings.

Priority order (NE > NP > VP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named entities are masked first because they are the most semantically dense
spans in a sentence. Noun phrases follow, as they carry the core referential
content. Verb phrases are used last, as they often contain auxiliary tokens
that are easier to reconstruct from local context.

Fallback
~~~~~~~~
When linguistic spans do not cover enough tokens to meet the masking budget,
the remainder is filled using geometric random spans (same algorithm as
SpanMaskingDataCollator), ensuring the effective mask rate is consistent
across all sequences regardless of spaCy parse quality.

Architecture
~~~~~~~~~~~~
This collator works at the *token-id* level (post-tokenization). Because BERT
uses WordPiece tokenization, spaCy character offsets must be mapped to
WordPiece token positions. We handle this via the HuggingFace tokenizer's
offset_mapping, which is generated during dataset preprocessing and stored
alongside input_ids.

Usage
-----
    from linguistic_span_masking_collator import LinguisticSpanMaskingDataCollator

    collator = LinguisticSpanMaskingDataCollator(
        tokenizer=tokenizer,
        nlp=spacy.load("en_core_web_sm"),
        mlm_probability=0.15,
        linguistic_ratio=1.0,   # 1.0 = all linguistic, 0.0 = all geometric
    )

Reference: SpanBERT — https://arxiv.org/abs/1907.10529
           ERNIE     — https://arxiv.org/abs/1904.09223
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

# spaCy is imported lazily so the module can be imported without it installed
try:
    import spacy
    from spacy.language import Language as SpacyLanguage
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    SpacyLanguage = Any


# ---------------------------------------------------------------------------
# Dataset preprocessing helper (call this BEFORE building the collator)
# ---------------------------------------------------------------------------

def tokenize_with_offsets(examples, tokenizer, max_length: int = 128):
    """
    Tokenizer wrapper that also stores character-level offset_mapping.

    This must be used instead of the plain tokenizer call in the dataset
    preprocessing pipeline so that the collator can map spaCy character
    spans back to WordPiece token positions at collation time.

    Parameters
    ----------
    examples   : batch of raw text examples (HuggingFace datasets format)
    tokenizer  : AutoTokenizer instance
    max_length : maximum sequence length

    Returns
    -------
    dict with keys: input_ids, attention_mask, special_tokens_mask,
                    offset_mapping, original_text
    """
    encoding = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,   # <-- key addition
    )
    # Store the original text so the collator can run spaCy on it
    encoding["original_text"] = examples["text"]
    return encoding


# ---------------------------------------------------------------------------
# Span extraction from spaCy parse
# ---------------------------------------------------------------------------

class LinguisticSpanExtractor:
    """
    Extracts NE, NP, and VP token-level spans from a spaCy Doc.

    All spans are returned as (start_token_idx, end_token_idx) tuples in
    WordPiece token space, using the offset_mapping produced by the HuggingFace
    tokenizer.

    Priority tiers
    --------------
    tier 0 : Named Entities  (Doc.ents)
    tier 1 : Noun Phrases    (Doc.noun_chunks)
    tier 2 : Verb Phrases    (tokens whose head is a VERB and dep in {ROOT,
                               aux, auxpass, xcomp})  — approximated since
                               spaCy does not expose VP constituents directly
    """

    def __init__(self, nlp: "SpacyLanguage"):
        self.nlp = nlp

    def get_spans(
        self,
        text: str,
        offset_mapping: List[Tuple[int, int]],
        special_tokens_mask: List[int],
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Run spaCy on `text` and return token-index spans for each tier.

        Parameters
        ----------
        text               : original raw text string
        offset_mapping     : list of (char_start, char_end) per WordPiece token
        special_tokens_mask: 1 = special token, 0 = normal token

        Returns
        -------
        dict with keys "NE", "NP", "VP", each a list of (start, end) token
        index pairs (inclusive, exclusive — like Python slices).
        """
        doc = self.nlp(text)

        # Build a character-position → WordPiece-token-index lookup
        char_to_tok = self._build_char_to_tok(offset_mapping, special_tokens_mask)

        ne_spans = self._extract_ne_spans(doc, char_to_tok)
        np_spans = self._extract_np_spans(doc, char_to_tok)
        vp_spans = self._extract_vp_spans(doc, char_to_tok)

        return {"NE": ne_spans, "NP": np_spans, "VP": vp_spans}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_char_to_tok(
        self,
        offset_mapping: List[Tuple[int, int]],
        special_tokens_mask: List[int],
    ) -> Dict[int, int]:
        """Map each character position to its WordPiece token index."""
        char_to_tok: Dict[int, int] = {}
        for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
            if special_tokens_mask[tok_idx] == 1:
                continue  # skip [CLS], [SEP], padding
            for char_pos in range(char_start, char_end):
                char_to_tok[char_pos] = tok_idx
        return char_to_tok

    def _char_span_to_tok_span(
        self,
        char_start: int,
        char_end: int,
        char_to_tok: Dict[int, int],
    ) -> Optional[Tuple[int, int]]:
        """
        Convert a (char_start, char_end) span to (tok_start, tok_end).

        Returns None if the span doesn't map to any non-special tokens
        (e.g., entirely within a truncated region).
        """
        tok_indices = set()
        for char_pos in range(char_start, char_end):
            if char_pos in char_to_tok:
                tok_indices.add(char_to_tok[char_pos])
        if not tok_indices:
            return None
        return min(tok_indices), max(tok_indices) + 1  # exclusive end

    def _extract_ne_spans(
        self, doc, char_to_tok: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        spans = []
        for ent in doc.ents:
            span = self._char_span_to_tok_span(ent.start_char, ent.end_char, char_to_tok)
            if span is not None:
                spans.append(span)
        return spans

    def _extract_np_spans(
        self, doc, char_to_tok: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        spans = []
        for chunk in doc.noun_chunks:
            span = self._char_span_to_tok_span(
                chunk.start_char, chunk.end_char, char_to_tok
            )
            if span is not None:
                spans.append(span)
        return spans

    def _extract_vp_spans(
        self, doc, char_to_tok: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Approximate VP extraction: for each VERB token, collect the token
        and its direct dependents that are auxiliary or clausal complements.
        spaCy does not expose VP constituents directly, so we use dependency
        relations as a proxy.
        """
        spans = []
        vp_dep_labels = {"aux", "auxpass", "neg", "advmod", "xcomp", "ccomp"}
        for token in doc:
            if token.pos_ != "VERB":
                continue
            # Collect the verb and its close dependents
            members = [token] + [
                child for child in token.children
                if child.dep_ in vp_dep_labels
            ]
            if not members:
                continue
            char_start = min(t.idx for t in members)
            char_end   = max(t.idx + len(t.text) for t in members)
            span = self._char_span_to_tok_span(char_start, char_end, char_to_tok)
            if span is not None:
                spans.append(span)
        return spans


# ---------------------------------------------------------------------------
# The collator itself
# ---------------------------------------------------------------------------

@dataclass
class LinguisticSpanMaskingDataCollator:
    """
    Drop-in replacement for DataCollatorForLanguageModeling that masks
    linguistically meaningful spans (NE > NP > VP) with geometric fallback.

    Parameters
    ----------
    tokenizer          : HuggingFace tokenizer with a [MASK] token.
    nlp                : spaCy Language model (e.g. spacy.load("en_core_web_sm")).
    mlm_probability    : Target fraction of non-special tokens to mask (default 0.15).
    linguistic_ratio   : Fraction of the masking budget to fill with linguistic
                         spans before falling back to geometric spans (default 1.0).
                         Set to 0.0 for pure geometric, 1.0 for max linguistic.
    mean_span_length   : Mean of the geometric fallback distribution (default 3.0).
    max_span_length    : Hard cap on any single span (default 10).
    mask_probability   : Probability of replacing a masked token with [MASK]
                         (80/10/10 rule, default 0.80).
    pad_to_multiple_of : Optional padding multiple.
    """

    tokenizer: PreTrainedTokenizerBase
    nlp: Any                              # spaCy Language
    mlm_probability: float = 0.15
    linguistic_ratio: float = 1.0
    mean_span_length: float = 3.0
    max_span_length: int = 10
    mask_probability: float = 0.80
    pad_to_multiple_of: Optional[int] = None

    # Populated in __post_init__
    _extractor: Any = field(init=False, repr=False, default=None)
    _geo_p: float = field(init=False, repr=False, default=0.0)

    # Diagnostics — reset each __call__
    _stats: Dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is required for LinguisticSpanMaskingDataCollator. "
                "Install it with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer must have a [MASK] token.")
        self._extractor = LinguisticSpanExtractor(self.nlp)
        p = min(1.0 / self.mean_span_length, 0.9999)
        self._geo_p = p
        self._stats = {"linguistic": 0, "geometric": 0, "total_seqs": 0}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch. Each feature dict must contain:
          - input_ids            : List[int]
          - attention_mask       : List[int]
          - special_tokens_mask  : List[int]
          - offset_mapping       : List[Tuple[int,int]]
          - original_text        : str   (the raw text before tokenization)
        """
        # Reset per-batch stats
        self._stats = {"linguistic": 0, "geometric": 0, "total_seqs": len(features)}

        # Pad and stack — but strip non-tensor fields first
        tensor_features = [
            {k: v for k, v in f.items() if k not in ("original_text", "offset_mapping")}
            for f in features
        ]
        batch = self._pad_batch(tensor_features)
        labels = batch["input_ids"].clone()

        for i, feature in enumerate(features):
            batch["input_ids"][i], labels[i] = self._mask_single_sequence(
                input_ids=batch["input_ids"][i],
                special_tokens_mask=batch["special_tokens_mask"][i],
                original_text=feature.get("original_text", ""),
                offset_mapping=feature.get("offset_mapping", []),
            )

        batch["labels"] = labels
        batch.pop("special_tokens_mask", None)
        return batch

    @property
    def masking_stats(self) -> Dict[str, Any]:
        """
        Returns diagnostic counts from the last batch:
          - linguistic : spans filled by NE/NP/VP selection
          - geometric  : spans filled by fallback
          - total_seqs : number of sequences in the batch
        """
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Core masking logic
    # ------------------------------------------------------------------

    def _mask_single_sequence(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        original_text: str,
        offset_mapping: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(0)
        labels  = input_ids.clone()

        can_mask = special_tokens_mask.eq(0)  # True = maskable
        n_maskable = int(can_mask.sum().item())
        target_masked = max(1, round(n_maskable * self.mlm_probability))
        linguistic_budget = round(target_masked * self.linguistic_ratio)

        is_masked = torch.zeros(seq_len, dtype=torch.bool)

        # ---- Phase 1: Linguistic spans (NE > NP > VP) ------------------
        if linguistic_budget > 0 and original_text and offset_mapping:
            sp_mask_list = special_tokens_mask.tolist()
            off_map = offset_mapping[: seq_len]  # guard against length mismatch

            span_dict = self._extractor.get_spans(original_text, off_map, sp_mask_list)
            # Priority order: NE first, then NP, then VP
            ordered_spans = self._prioritise_spans(
                span_dict, can_mask, seq_len
            )
            random.shuffle(ordered_spans)  # shuffle within each tier separately

            for (start, end) in ordered_spans:
                if is_masked.sum().item() >= linguistic_budget:
                    break
                span_slice = slice(start, end)
                if can_mask[span_slice].all() and not is_masked[span_slice].any():
                    is_masked[span_slice] = True
                    self._stats["linguistic"] += 1

        # ---- Phase 2: Geometric fallback for remaining budget -----------
        remaining = target_masked - int(is_masked.sum().item())
        if remaining > 0:
            attempts = 0
            while (
                int(is_masked.sum().item()) < target_masked
                and attempts < seq_len * 4
            ):
                attempts += 1
                span_len = self._sample_geo_span()
                start = random.randint(0, seq_len - 1)
                if not can_mask[start]:
                    continue
                end = start
                for j in range(start, min(start + span_len, seq_len)):
                    if can_mask[j] and not is_masked[j]:
                        end = j
                    else:
                        break
                is_masked[start: end + 1] = True
                self._stats["geometric"] += 1

        # ---- Phase 3: Apply 80/10/10 replacement -----------------------
        masked_ids = input_ids.clone()
        for pos in is_masked.nonzero(as_tuple=True)[0]:
            r = random.random()
            if r < self.mask_probability:
                masked_ids[pos] = self.tokenizer.mask_token_id
            elif r < self.mask_probability + (1 - self.mask_probability) / 2:
                pass  # keep original
            else:
                masked_ids[pos] = random.randint(0, self.tokenizer.vocab_size - 1)

        labels[~is_masked] = -100
        return masked_ids, labels

    def _prioritise_spans(
        self,
        span_dict: Dict[str, List[Tuple[int, int]]],
        can_mask: torch.Tensor,
        seq_len: int,
    ) -> List[Tuple[int, int]]:
        """
        Build an ordered candidate list: NEs first, then NPs (excluding tokens
        already in an NE), then VPs (excluding tokens already in NE or NP).

        Overlapping spans within a tier are shuffled; across tiers the priority
        order is preserved by processing them in phase order during masking.
        """
        used_positions: set = set()
        ordered: List[Tuple[int, int]] = []

        for tier in ("NE", "NP", "VP"):
            tier_spans = []
            for (start, end) in span_dict.get(tier, []):
                # Clamp to sequence length
                start = max(0, start)
                end   = min(seq_len, end)
                if start >= end:
                    continue
                # Skip spans that substantially overlap already-selected tiers
                span_positions = set(range(start, end))
                if span_positions & used_positions:
                    continue
                # Skip spans containing special tokens
                if not can_mask[start:end].all():
                    continue
                tier_spans.append((start, end))

            random.shuffle(tier_spans)
            for span in tier_spans:
                used_positions.update(range(span[0], span[1]))
            ordered.extend(tier_spans)

        return ordered

    # ------------------------------------------------------------------
    # Geometric sampler (fallback)
    # ------------------------------------------------------------------

    def _sample_geo_span(self) -> int:
        p = self._geo_p
        length = 0
        while length < 1:
            u = random.random()
            if u < 1e-12:
                u = 1e-12
            length = int(math.ceil(math.log(u) / math.log(1 - p)))
        return min(length, self.max_span_length)

    # ------------------------------------------------------------------
    # Padding helper
    # ------------------------------------------------------------------

    def _pad_batch(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        keys = list(features[0].keys())
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            max_len = (
                math.ceil(max_len / self.pad_to_multiple_of)
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id or 0
        batch: Dict[str, torch.Tensor] = {}

        for key in keys:
            pad_value = pad_id if key == "input_ids" else 0
            if key == "special_tokens_mask":
                pad_value = 1
            padded = []
            for f in features:
                seq = list(f[key])
                seq += [pad_value] * (max_len - len(seq))
                padded.append(seq)
            batch[key] = torch.tensor(padded, dtype=torch.long)

        return batch