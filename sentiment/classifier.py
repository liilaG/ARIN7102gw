"""Document-level sentiment classifier using FinBERT models.

Supports bilingual routing:
  - Chinese (zh)  -> yiyanghkust/finbert-tone-chinese
  - English (en)  -> ProsusAI/finbert
  - Mixed/unknown -> both models, scores averaged

Device selection: CUDA > MPS > CPU.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .schemas import Language, PreprocessedDoc, SentimentItem, TextLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_IDS: dict[str, str] = {
    "zh": "yiyanghkust/finbert-tone-chinese",
    "en": "ProsusAI/finbert",
}

# Label mapping: model-specific -> unified positive/negative/neutral
_LABEL_MAP: dict[str, dict[str, str]] = {
    "zh": {"LABEL_0": "neutral", "LABEL_1": "positive", "LABEL_2": "negative"},
    "en": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
}

# Score direction: 0=negative, 0.5=neutral, 1.0=positive
_LABEL_SCORE: dict[str, float] = {
    "positive": 0.85,
    "neutral": 0.50,
    "negative": 0.15,
}


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class SentimentClassifier:
    """Document-level financial sentiment classifier using FinBERT models."""

    def __init__(self, device: str | None = None, max_sentences: int = 20):
        self._device = device or _detect_device()
        self._max_sentences = max_sentences
        self._models: dict[str, Any] = {}  # lang -> pipeline
        logger.info("SentimentClassifier device=%s max_sentences=%s", self._device, self._max_sentences)

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self, language: str) -> Any:
        """Load the FinBERT pipeline for a given language (lazy, cached)."""
        if language in self._models:
            return self._models[language]

        model_id = _MODEL_IDS.get(language)
        if model_id is None:
            raise ValueError(f"Unsupported language: {language}")

        logger.info("Loading %s model: %s", language, model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(self._device)
        model.eval()

        self._models[language] = (tokenizer, model)
        return self._models[language]

    def _infer_single(self, text: str, language: str) -> dict[str, float]:
        """Run inference on a single text and return label→score dict."""
        if not text.strip():
            return {"neutral": 1.0}

        tokenizer, model = self._load_model(language)
        label_map = _LABEL_MAP.get(language, {})

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self._device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()

        # Map model-specific labels to unified labels
        id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
        result: dict[str, float] = {}
        for i, score in enumerate(probs):
            raw_label = id2label.get(i, f"LABEL_{i}")
            unified = label_map.get(raw_label, raw_label.lower())
            result[unified] = result.get(unified, 0.0) + score
        return result

    def _predict_label_and_score(self, text: str, language: Language) -> tuple[str, float, float]:
        """Predict (sentiment_label, sentiment_score, confidence) for text."""
        if language in ("zh", "en"):
            scores = self._infer_single(text, language)
        else:
            # mixed / unknown: run both models and average
            scores_zh = self._infer_single(text, "zh") if language != "en" else {}
            scores_en = self._infer_single(text, "en") if language != "zh" else {}
            combined: dict[str, float] = {}
            for k in set(scores_zh) | set(scores_en):
                combined[k] = (scores_zh.get(k, 0) + scores_en.get(k, 0)) / 2
            scores = combined

        if not scores:
            return "neutral", 0.5, 0.0

        # Sort by descending probability
        sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        label, confidence = sorted_labels[0]

        # sentiment_score: weighted average mapped to 0-1 scale
        score = sum(_LABEL_SCORE.get(l, 0.5) * s for l, s in scores.items())
        # Normalize: divide by sum of weights
        total_s = sum(scores.values()) or 1.0
        score = score / total_s

        return label, round(score, 4), round(confidence, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_document(self, doc: PreprocessedDoc) -> SentimentItem:
        """Classify a preprocessed document sentence-by-sentence, then aggregate.

        Each sentence is scored individually (FinBERT is sentence-level).
        Results are aggregated by majority label and average score.
        """
        sentences = [s for s in doc.sentences if s.strip()] if doc.sentences else [doc.raw_text]
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return _empty_sentiment(doc)

        # Cap at max_sentences for performance on very long documents
        if len(sentences) > self._max_sentences:
            logger.debug(
                "Truncating doc %s from %d to %d sentences",
                doc.evidence_id, len(sentences), self._max_sentences,
            )
            sentences = sentences[:self._max_sentences]

        per_sentence: list[tuple[str, float, float]] = []
        for sent in sentences:
            label, score, conf = self._predict_label_and_score(sent, doc.language)
            per_sentence.append((label, score, conf))

        # Aggregate: majority label, average score, average confidence
        label_counts = Counter(l for l, _, _ in per_sentence)
        majority_label = label_counts.most_common(1)[0][0]
        avg_score = sum(s for _, s, _ in per_sentence) / len(per_sentence)
        avg_conf = sum(c for _, _, c in per_sentence) / len(per_sentence)

        return SentimentItem(
            evidence_id=doc.evidence_id,
            source_type=doc.source_type,
            title=doc.title,
            publish_time=doc.publish_time,
            source_name=doc.source_name,
            entity_symbols=doc.entity_hits,
            sentiment_label=majority_label,
            sentiment_score=round(avg_score, 4),
            confidence=round(avg_conf, 4),
            text_level=doc.text_level,
            relevant_excerpt=" ".join(sentences),
            rank_score=doc.rank_score,
        )

    def analyze_documents(self, docs: list[PreprocessedDoc]) -> list[SentimentItem]:
        """Classify a batch of preprocessed documents."""
        results: list[SentimentItem] = []
        for doc in docs:
            if doc.skipped:
                continue
            try:
                results.append(self.analyze_document(doc))
            except Exception:
                logger.exception("Error analyzing document %s", doc.evidence_id)
        return results

    # ------------------------------------------------------------------
    # Convenience: end-to-end from Preprocessor output
    # ------------------------------------------------------------------

    def analyze_pipeline_output(
        self, processed_docs: list[PreprocessedDoc],
    ) -> list[SentimentItem]:
        """Thin wrapper alias for analyze_documents."""
        return self.analyze_documents(processed_docs)


def _empty_sentiment(doc: PreprocessedDoc) -> SentimentItem:
    """Return a neutral placeholder when no text is available."""
    return SentimentItem(
        evidence_id=doc.evidence_id,
        source_type=doc.source_type,
        title=doc.title,
        publish_time=doc.publish_time,
        source_name=doc.source_name,
        entity_symbols=doc.entity_hits,
        sentiment_label="neutral",
        sentiment_score=0.5,
        confidence=0.0,
        text_level=doc.text_level,
        rank_score=doc.rank_score,
    )
