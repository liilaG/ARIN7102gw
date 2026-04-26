"""Integration tests for the full sentiment pipeline: Preprocessor → SentimentClassifier.

Tests the public API boundary as exported by sentiment/__init__.py.
Uses mocked inference to avoid downloading FinBERT models in CI.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentiment import (
    FilterMeta,
    PreprocessedDoc,
    Preprocessor,
    QueryInput,
    SentimentClassifier,
    SentimentItem,
)
from sentiment.preprocessor import normalize_text


# ===========================================================================
# Shared test data
# ===========================================================================

NLU_STOCK = {
    "product_type": {"label": "stock"},
    "entities": [
        {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
        {"symbol": "000858.SZ", "canonical_name": "五粮液", "mention": "五粮液"},
    ],
}

DOC_BODY_NEWS = {
    "evidence_id": "news_001",
    "source_type": "news",
    "source_name": "证券时报",
    "publish_time": "2026-04-23T20:17:00",
    "title": "贵州茅台营收增长",
    "summary": "茅台业绩超预期",
    "body": "贵州茅台今日发布公告。公司营收增长显著。五粮液也有不错表现。",
    "body_available": True,
    "rank_score": 0.92,
}

DOC_SHORT_ANNOUNCEMENT = {
    "evidence_id": "ann_002",
    "source_type": "announcement",
    "source_name": "深交所",
    "publish_time": "2026-04-22T10:00:00",
    "title": "五粮液发布最新财报",
    "summary": "五粮液发布最新财报",
    "body_available": False,
    "rank_score": 0.88,
}

DOC_RESEARCH_NOTE = {
    "evidence_id": "research_003",
    "source_type": "research_note",
    "source_name": "某券商研究所",
    "publish_time": "2026-04-21T14:30:00",
    "title": "Apple Q1 earnings analysis",
    "summary": "Strong revenue growth across all segments",
    "body": "Apple Inc. reported record quarterly earnings. Revenue grew 15% year over year. The iPhone segment outperformed expectations significantly.",
    "body_available": True,
    "rank_score": 0.75,
}

DOC_PRODUCT_DOC = {
    "evidence_id": "pdoc_004",
    "source_type": "product_doc",
    "title": "茅台介绍",
    "body": "茅台是高端白酒龙头。产品线丰富多样。",
    "body_available": True,
    "rank_score": 0.60,
}

DOC_UNSUPPORTED = {
    "evidence_id": "faq_005",
    "source_type": "faq",
    "title": "什么是ETF？",
}

RETRIEVAL_SAMPLE = {
    "documents": [DOC_BODY_NEWS, DOC_SHORT_ANNOUNCEMENT, DOC_RESEARCH_NOTE, DOC_PRODUCT_DOC, DOC_UNSUPPORTED],
}


# ===========================================================================
# Helpers
# ===========================================================================


def _mock_infer_scores(text: str, language: str) -> dict[str, float]:
    """Deterministic mock: base sentiment on keyword presence."""
    positive_keywords = ["增长", "revenue grew", "record", "龙头", "不错", "超预期"]
    negative_keywords = ["下降", "decline", "loss"]

    is_positive = any(kw in text for kw in positive_keywords)
    is_negative = any(kw in text for kw in negative_keywords)

    if is_positive and not is_negative:
        return {"positive": 0.85, "neutral": 0.10, "negative": 0.05}
    elif is_negative and not is_positive:
        return {"positive": 0.05, "neutral": 0.15, "negative": 0.80}
    else:
        return {"positive": 0.10, "neutral": 0.80, "negative": 0.10}


# ===========================================================================
# End-to-end pipeline tests (with mocked inference)
# ===========================================================================


class TestSentimentPipeline:
    """Full Preprocessor → SentimentClassifier integration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.preprocessor = Preprocessor()
        self.classifier = SentimentClassifier(device="cpu")

    def test_full_pipeline_integration(self):
        """Preprocess then classify: end-to-end with deterministic mock."""
        skip_reason, docs, meta = self.preprocessor.process_query(NLU_STOCK, RETRIEVAL_SAMPLE)

        assert skip_reason is None
        assert meta.analyzed_docs_count == 4  # news + announcement + research_note + product_doc
        assert meta.skipped_docs_count == 1    # faq
        assert meta.short_text_fallback_count == 1  # announcement (no body)

        with patch.object(self.classifier, "_infer_single", side_effect=_mock_infer_scores):
            results = self.classifier.analyze_documents(docs)

        assert len(results) == 4  # skipped faq excluded
        assert all(isinstance(r, SentimentItem) for r in results)

        # news_001: Chinese text with positive keywords → positive
        r_news = next(r for r in results if r.evidence_id == "news_001")
        assert r_news.sentiment_label == "positive"
        assert r_news.sentiment_score > 0.6
        assert r_news.source_type == "news"
        assert r_news.title == "贵州茅台营收增长"
        assert "600519.SH" in r_news.entity_symbols

        # ann_002: short text (no body), title+summary fallback
        r_ann = next(r for r in results if r.evidence_id == "ann_002")
        assert r_ann.text_level == "short"
        assert r_ann.sentiment_label in ("positive", "negative", "neutral")

        # research_003: English text, Apple earnings
        r_en = next(r for r in results if r.evidence_id == "research_003")
        assert r_en.source_type == "research_note"
        assert r_en.sentiment_label in ("positive", "negative", "neutral")

        # pdoc_004: product_doc with positive keywords
        r_pdoc = next(r for r in results if r.evidence_id == "pdoc_004")
        assert r_pdoc.source_type == "product_doc"

    def test_pipeline_preserves_rank_order(self):
        """Documents should be processed in order; results maintain the input sequence."""
        skip_reason, docs, meta = self.preprocessor.process_query(NLU_STOCK, RETRIEVAL_SAMPLE)
        with patch.object(self.classifier, "_infer_single", side_effect=_mock_infer_scores):
            results = self.classifier.analyze_documents(docs)

        evidence_ids = [r.evidence_id for r in results]
        assert evidence_ids == ["news_001", "ann_002", "research_003", "pdoc_004"]

    def test_pipeline_skip_by_product_type(self):
        """Queries with out_of_scope product type produce empty results."""
        nlu = {"product_type": {"label": "out_of_scope"}, "entities": []}
        skip_reason, docs, meta = self.preprocessor.process_query(nlu, {"documents": []})

        assert skip_reason == "product_type=out_of_scope"
        assert docs == []
        assert meta.skipped_by_product_type is True

        results = self.classifier.analyze_documents(docs)
        assert results == []

    def test_pipeline_empty_documents(self):
        """No documents in retrieval result → empty output."""
        skip_reason, docs, meta = self.preprocessor.process_query(NLU_STOCK, {"documents": []})
        assert skip_reason is None
        assert docs == []
        assert meta.analyzed_docs_count == 0

        results = self.classifier.analyze_documents(docs)
        assert results == []

    def test_pipeline_error_isolation_full_chain(self):
        """A bad document should not crash the entire pipeline."""
        bad_doc = {
            "evidence_id": "bad_001",
            "source_type": "news",
            "title": "Bad",
            "body": "valid text but will be fine in preprocessor",
            "body_available": True,
        }
        retrieval = {"documents": [bad_doc, DOC_BODY_NEWS]}

        # Make the classifier fail on a specific doc
        skip_reason, docs, meta = self.preprocessor.process_query(NLU_STOCK, retrieval)
        assert meta.analyzed_docs_count == 2

        def _failing_infer(text: str, language: str) -> dict[str, float]:
            if "贵州茅台" in text:
                raise RuntimeError("inference failure")
            return _mock_infer_scores(text, language)

        with patch.object(self.classifier, "_infer_single", side_effect=_failing_infer):
            results = self.classifier.analyze_documents(docs)

        # The good doc (bad_001) should still be in results; news_001 should be excluded
        assert len(results) == 1
        assert results[0].evidence_id == "bad_001"

    def test_pipeline_analyze_pipeline_output_alias(self):
        """analyze_pipeline_output is an alias for analyze_documents."""
        skip_reason, docs, meta = self.preprocessor.process_query(NLU_STOCK, RETRIEVAL_SAMPLE)
        with patch.object(self.classifier, "_infer_single", side_effect=_mock_infer_scores):
            r1 = self.classifier.analyze_documents(docs)
            r2 = self.classifier.analyze_pipeline_output(docs)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.evidence_id == b.evidence_id
            assert a.sentiment_label == b.sentiment_label

    def test_pipeline_mixed_language_docs(self):
        """Pipeline handles a batch with mixed zh/en docs."""
        retrieval = {"documents": [DOC_BODY_NEWS, DOC_RESEARCH_NOTE]}

        skip_reason, docs, meta = self.preprocessor.process_query(NLU_STOCK, retrieval)
        assert meta.analyzed_docs_count == 2

        zh_doc = next(d for d in docs if d.evidence_id == "news_001")
        en_doc = next(d for d in docs if d.evidence_id == "research_003")
        assert zh_doc.language == "zh"
        assert en_doc.language == "en"

        with patch.object(self.classifier, "_infer_single", side_effect=_mock_infer_scores):
            results = self.classifier.analyze_documents(docs)

        assert len(results) == 2

    def test_pipeline_sentiment_scores_are_bounded(self):
        """Sentiment scores must be in [0, 1] range."""
        skip_reason, docs, meta = self.preprocessor.process_query(NLU_STOCK, RETRIEVAL_SAMPLE)
        with patch.object(self.classifier, "_infer_single", side_effect=_mock_infer_scores):
            results = self.classifier.analyze_documents(docs)

        for r in results:
            assert 0.0 <= r.sentiment_score <= 1.0, f"{r.evidence_id}: score={r.sentiment_score}"
            assert 0.0 <= r.confidence <= 1.0, f"{r.evidence_id}: conf={r.confidence}"
            assert r.sentiment_label in ("positive", "negative", "neutral")


# ===========================================================================
# FilterMeta passthrough tests
# ===========================================================================


class TestFilterMetaPassthrough:
    """Verify FilterMeta correctly summarizes preprocessing results."""

    def test_filter_meta_counts_all_supported(self):
        preprocessor = Preprocessor()
        retrieval = {
            "documents": [DOC_BODY_NEWS, DOC_RESEARCH_NOTE, DOC_PRODUCT_DOC],
        }
        _, __, meta = preprocessor.process_query(NLU_STOCK, retrieval)
        assert meta.analyzed_docs_count == 3
        assert meta.skipped_docs_count == 0
        assert meta.skipped_by_product_type is False
        assert meta.skipped_by_intent is False

    def test_filter_meta_mixed_batch(self):
        preprocessor = Preprocessor()
        _, __, meta = preprocessor.process_query(NLU_STOCK, RETRIEVAL_SAMPLE)
        assert meta.analyzed_docs_count == 4
        assert meta.skipped_docs_count == 1
        assert meta.short_text_fallback_count == 1

    def test_filter_meta_all_unsupported(self):
        preprocessor = Preprocessor()
        retrieval = {"documents": [DOC_UNSUPPORTED, DOC_UNSUPPORTED]}
        _, __, meta = preprocessor.process_query(NLU_STOCK, retrieval)
        assert meta.analyzed_docs_count == 0
        assert meta.skipped_docs_count == 2


# ===========================================================================
# QueryInput model tests
# ===========================================================================


class TestQueryInputModel:
    """Validate the QueryInput pydantic model."""

    def test_query_input_valid_stock(self):
        qi = QueryInput(
            query_id="q_001",
            product_type_label="stock",
            intent_labels=["sentiment_analysis"],
            topic_labels=["equity"],
            entities=[{"symbol": "600519.SH", "canonical_name": "贵州茅台"}],
            documents=[DOC_BODY_NEWS, DOC_SHORT_ANNOUNCEMENT],
        )
        assert qi.query_id == "q_001"
        assert qi.product_type_label == "stock"
        assert len(qi.documents) == 2

    def test_query_input_empty_docs(self):
        qi = QueryInput(
            query_id="q_002",
            product_type_label="macro",
            intent_labels=[],
            topic_labels=[],
            entities=[],
            documents=[],
        )
        assert qi.documents == []

    def test_query_input_roundtrip(self):
        """QueryInput can be built from real NLU + retrieval dicts."""
        qi = QueryInput(
            query_id="q_003",
            product_type_label=NLU_STOCK["product_type"]["label"],
            intent_labels=["research"],
            topic_labels=["equity"],
            entities=NLU_STOCK["entities"],
            documents=RETRIEVAL_SAMPLE["documents"],
        )
        # Should not raise
        assert qi.product_type_label == "stock"
        assert len(qi.entities) == 2
        assert len(qi.documents) == 5


# ===========================================================================
# Serialization round-trip
# ===========================================================================


class TestSerializationRoundTrip:
    """Pydantic model JSON serialization round-trips."""

    def test_preprocessed_doc_roundtrip(self):
        doc = PreprocessedDoc(
            evidence_id="test_001",
            source_type="news",
            title="测试标题",
            raw_text="测试内容。",
            language="zh",
            sentences=["测试内容。"],
            entity_hits=["600519.SH"],
            text_level="full",
            relevant_excerpt="测试内容。",
        )
        json_str = doc.model_dump_json()
        restored = PreprocessedDoc.model_validate_json(json_str)
        assert restored.evidence_id == doc.evidence_id
        assert restored.language == "zh"
        assert restored.entity_hits == ["600519.SH"]

    def test_sentiment_item_roundtrip(self):
        item = SentimentItem(
            evidence_id="test_001",
            source_type="news",
            title="测试",
            sentiment_label="positive",
            sentiment_score=0.85,
            confidence=0.92,
            text_level="full",
            entity_symbols=["600519.SH"],
        )
        json_str = item.model_dump_json()
        restored = SentimentItem.model_validate_json(json_str)
        assert restored.sentiment_label == "positive"
        assert restored.sentiment_score == 0.85
        assert restored.confidence == 0.92

    def test_filter_meta_roundtrip(self):
        meta = FilterMeta(
            skipped_by_product_type=False,
            skipped_by_intent=False,
            skipped_docs_count=2,
            short_text_fallback_count=1,
            analyzed_docs_count=10,
        )
        json_str = meta.model_dump_json()
        restored = FilterMeta.model_validate_json(json_str)
        assert restored.skipped_docs_count == 2
        assert restored.analyzed_docs_count == 10

    def test_query_input_roundtrip(self):
        qi = QueryInput(
            query_id="q_serial",
            product_type_label="stock",
            intent_labels=["research"],
            topic_labels=[],
            entities=[{"symbol": "600519.SH", "canonical_name": "贵州茅台"}],
            documents=[DOC_BODY_NEWS],
        )
        json_str = qi.model_dump_json()
        restored = QueryInput.model_validate_json(json_str)
        assert restored.query_id == qi.query_id
        assert len(restored.documents) == 1


# ===========================================================================
# Sentence cap with mocked pipeline
# ===========================================================================


class TestSentenceCapPipeline:
    """max_sentences capping works correctly in the full pipeline."""

    def test_long_document_is_capped(self):
        preprocessor = Preprocessor()
        classifier = SentimentClassifier(device="cpu", max_sentences=3)

        # Build a doc with many sentence-ending punctuation marks
        sentences = [f"这是第{i}句话。" for i in range(10)]
        long_body = "".join(sentences)
        long_doc = {
            "evidence_id": "long_001",
            "source_type": "news",
            "title": "长文档",
            "body": long_body,
            "body_available": True,
        }

        skip_reason, docs, _ = preprocessor.process_query(NLU_STOCK, retrieval_result={"documents": [long_doc]})
        assert len(docs[0].sentences) >= 8  # should split into many sentences

        with patch.object(classifier, "_infer_single", side_effect=_mock_infer_scores):
            results = classifier.analyze_documents(docs)

        assert len(results) == 1
        excerpt = results[0].relevant_excerpt or ""
        # Only max_sentences (3) sentences in the excerpt
        assert excerpt.count("句话") == 3


# ===========================================================================
# Demo data fixture tests (data/sentiment_demo_*.json)
# ===========================================================================


class TestDemoDataFixtures:
    """Verify the demo data files are loadable and pipeline-ready."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.preprocessor = Preprocessor()
        self.classifier = SentimentClassifier(device="cpu")

    def _load_demo(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        nlu = json.loads((root / "data" / "sentiment_demo_nlu.json").read_text(encoding="utf-8"))
        retrieval = json.loads((root / "data" / "sentiment_demo_retrieval.json").read_text(encoding="utf-8"))
        return nlu, retrieval

    def test_demo_data_loadable(self):
        """Demo data files exist and are valid JSON."""
        nlu, retrieval = self._load_demo()
        assert nlu["product_type"]["label"] == "stock"
        assert len(nlu["entities"]) == 4
        assert len(retrieval["documents"]) == 11

    def test_demo_data_preprocessing(self):
        """Preprocessing runs on demo data without errors."""
        nlu, retrieval = self._load_demo()
        skip_reason, docs, meta = self.preprocessor.process_query(nlu, retrieval)
        assert skip_reason is None
        assert meta.analyzed_docs_count == 10
        assert meta.skipped_docs_count == 1       # faq
        assert meta.short_text_fallback_count == 1  # demo_news_short_no_body
        assert len(docs) == 11

        skipped = [d for d in docs if d.skipped]
        assert len(skipped) == 1
        assert skipped[0].evidence_id == "demo_faq_unsupported"

        # Verify language detection
        zh_docs = [d for d in docs if d.language == "zh"]
        en_docs = [d for d in docs if d.language == "en"]
        assert len(zh_docs) >= 6
        assert len(en_docs) >= 2

    def test_demo_data_full_pipeline(self):
        """Full Preprocessor → Classifier pipeline runs on demo data."""
        nlu, retrieval = self._load_demo()
        skip_reason, docs, _ = self.preprocessor.process_query(nlu, retrieval)
        assert skip_reason is None

        with patch.object(self.classifier, "_infer_single", side_effect=_mock_infer_scores):
            results = self.classifier.analyze_documents(docs)

        assert len(results) == 10  # faq excluded
        assert all(isinstance(r, SentimentItem) for r in results)

        labels = {r.sentiment_label for r in results}
        assert labels.issubset({"positive", "negative", "neutral"})

    def test_demo_data_source_type_coverage(self):
        """Demo data covers all supported source types."""
        _, retrieval = self._load_demo()
        source_types = {d["source_type"] for d in retrieval["documents"]}
        assert source_types >= {"news", "announcement", "research_note", "product_doc"}
        assert "faq" in source_types  # unsupported, should be skipped

    def test_demo_data_language_coverage(self):
        """Demo data includes Chinese, English, and mixed content."""
        nlu, retrieval = self._load_demo()
        _, docs, _ = self.preprocessor.process_query(nlu, retrieval)
        languages = {d.language for d in docs if not d.skipped}
        assert "zh" in languages
        assert "en" in languages
