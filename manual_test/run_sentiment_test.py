from __future__ import annotations

"""Manual test script for the Document Sentiment Analysis pipeline.

Usage:
  python manual_test/run_sentiment_test.py
      Run with built-in demo data (mock inference, no model download).

  python manual_test/run_sentiment_test.py --real-models
      Download and use real FinBERT models (takes several minutes first run).

  python manual_test/run_sentiment_test.py --input manual_test/output/<run-dir>
      Use NLU + retrieval JSON files from a previous run_manual_query.py run.

  python manual_test/run_sentiment_test.py --json-file data.json
      Use a custom JSON file containing {"nlu_result": {...}, "retrieval_result": {...}}.

Output is written to manual_test/output/<timestamp>-sentiment-<slug>/
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment import Preprocessor, SentimentClassifier

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ===========================================================================
# Built-in demo data
# ===========================================================================

DEMO_NLU = {
    "product_type": {"label": "stock"},
    "entities": [
        {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
        {"symbol": "000858.SZ", "canonical_name": "五粮液", "mention": "五粮液"},
    ],
}

DEMO_RETRIEVAL = {
    "documents": [
        {
            "evidence_id": "demo_news_001",
            "source_type": "news",
            "source_name": "证券时报",
            "publish_time": "2026-04-23T20:17:00",
            "title": "贵州茅台一季度营收增长超预期",
            "summary": "贵州茅台发布一季报，营收同比增长20%",
            "body": (
                "贵州茅台今日发布2026年一季度财报。"
                "公司营收达到450亿元，同比增长20%，超出市场预期。"
                "净利润率达到52%，创历史新高。"
                "分析师认为茅台品牌护城河深厚，长期看好。"
            ),
            "body_available": True,
            "rank_score": 0.95,
        },
        {
            "evidence_id": "demo_news_002",
            "source_type": "news",
            "source_name": "财经网",
            "publish_time": "2026-04-23T21:00:00",
            "title": "五粮液股价下跌分析",
            "summary": "五粮液近期股价持续走低",
            "body": (
                "五粮液股价近日连续下跌。"
                "市场担忧白酒行业增速放缓。"
                "经销商库存水平有所上升。"
                "但公司基本面仍然稳健，长期投资价值存在。"
            ),
            "body_available": True,
            "rank_score": 0.88,
        },
        {
            "evidence_id": "demo_ann_003",
            "source_type": "announcement",
            "source_name": "上交所",
            "publish_time": "2026-04-22T15:30:00",
            "title": "贵州茅台分红公告",
            "summary": "贵州茅台公布2025年度分红方案",
            "body": (
                "贵州茅台董事会审议通过2025年度利润分配方案。"
                "拟向全体股东每10股派发现金红利220元。"
                "分红总额创历史新高，彰显公司对股东的回报意愿。"
            ),
            "body_available": True,
            "rank_score": 0.82,
        },
        {
            "evidence_id": "demo_research_004",
            "source_type": "research_note",
            "source_name": "某券商研究所",
            "publish_time": "2026-04-21T10:00:00",
            "title": "Apple Inc. Q1 2026 Earnings Review",
            "summary": "Record revenue and strong iPhone sales",
            "body": (
                "Apple Inc. reported record quarterly revenue of $95 billion. "
                "iPhone revenue grew 12% year-over-year, driven by strong demand in emerging markets. "
                "Services revenue reached an all-time high. "
                "However, gross margins declined slightly due to component cost increases."
            ),
            "body_available": True,
            "rank_score": 0.78,
        },
        {
            "evidence_id": "demo_faq_005",
            "source_type": "faq",
            "title": "什么是ETF？",
            "summary": "交易型开放式指数基金",
        },
    ],
}


# ===========================================================================
# Mock inference (for --dry-run / default mode)
# ===========================================================================


def _mock_infer(text: str, language: str) -> dict[str, float]:
    """Deterministic mock based on keyword presence (no model needed)."""
    positive_kw = [
        "增长", "revenue grew", "record", "龙头", "不错", "超预期", "创历史",
        "看好", "新高", "回报", "分红", "稳健", "突破", "突破性", "祝贺",
        "嘉奖", "里程碑", "创新", "升级", "supercycle", "stellar",
        "overweight", "beat", "上调", "固态电池", "供不应求", "前景",
    ]
    negative_kw = [
        "下降", "decline", "loss", "下跌", "走低", "担忧", "放缓",
        "miss", "downgrade", "sell", "collapse", "headwind", "scrutiny",
        "蒸发", "压力", "下调", "寒流", "积压", "松动", "不确定性",
        "不确定", "严峻", "延迟", "阻力", "受阻", "不容乐观",
    ]

    pos = any(kw in text.casefold() for kw in positive_kw)
    neg = any(kw in text.casefold() for kw in negative_kw)

    if pos and not neg:
        return {"positive": 0.85, "neutral": 0.10, "negative": 0.05}
    elif neg and not pos:
        return {"positive": 0.05, "neutral": 0.15, "negative": 0.80}
    elif pos and neg:
        return {"positive": 0.30, "neutral": 0.50, "negative": 0.20}
    else:
        return {"positive": 0.10, "neutral": 0.80, "negative": 0.10}


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    args = _build_parser().parse_args()

    nlu_result, retrieval_result = _load_data(args)

    if args.mock:
        classifier = SentimentClassifier(device="cpu", max_sentences=20)
        _patch_infer(classifier)
        print("Using mock inference (no real models loaded).")
    else:
        print("Loading FinBERT models (this may take a while on first run)...")
        classifier = SentimentClassifier(max_sentences=20)

    preprocessor = Preprocessor()

    # Step 1: Preprocess
    print("\n--- Step 1: Preprocessing ---")
    skip_reason, docs, meta = preprocessor.process_query(nlu_result, retrieval_result)

    print(f"  query skip: {skip_reason}")
    print(f"  analyzed:   {meta.analyzed_docs_count}")
    print(f"  skipped:    {meta.skipped_docs_count}")
    print(f"  short:      {meta.short_text_fallback_count}")

    for doc in docs:
        flag = " [SKIPPED]" if doc.skipped else ""
        print(f"    {doc.evidence_id} | {doc.source_type} | {doc.language} | {doc.text_level} | {len(doc.sentences)} sentences{flag}")

    # Step 2: Classify
    print("\n--- Step 2: Sentiment Classification ---")
    results = classifier.analyze_documents(docs)

    for item in results:
        bar = _score_bar(item.sentiment_score)
        print(
            f"    {item.evidence_id:20s} | {item.sentiment_label:8s} | "
            f"score={item.sentiment_score:.2f} conf={item.confidence:.2f} | {bar}"
        )

    # Write outputs
    run_dir = _prepare_run_dir("sentiment-demo")
    (run_dir / "nlu_input.json").write_text(
        json.dumps(nlu_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "retrieval_input.json").write_text(
        json.dumps(retrieval_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    preprocessed_json = [
        {**doc.model_dump(), "sentences_count": len(doc.sentences)}
        for doc in docs
    ]
    (run_dir / "preprocessed_docs.json").write_text(
        json.dumps(preprocessed_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sentiment_json = [item.model_dump() for item in results]
    (run_dir / "sentiment_results.json").write_text(
        json.dumps(sentiment_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary JSON
    summary = {
        "mode": "mock" if args.mock else "real_models",
        "query_skip_reason": skip_reason,
        "filter_meta": meta.model_dump(),
        "doc_count": len(docs),
        "result_count": len(results),
        "sentiment_summary": {
            "positive": sum(1 for r in results if r.sentiment_label == "positive"),
            "negative": sum(1 for r in results if r.sentiment_label == "negative"),
            "neutral": sum(1 for r in results if r.sentiment_label == "neutral"),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nOutput written to: {run_dir}")


# ===========================================================================
# Helpers
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual test: Document Sentiment Analysis pipeline"
    )
    parser.add_argument(
        "--real-models",
        action="store_false",
        dest="mock",
        help="Download and use real FinBERT models instead of mock inference.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        metavar="DIR",
        help="Read nlu_result.json + retrieval_result.json from a previous manual query run directory.",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="",
        metavar="FILE",
        help="Read a single JSON file with nlu_result + retrieval_result keys.",
    )
    parser.add_argument(
        "--nlu-file",
        type=str,
        default="",
        metavar="FILE",
        help="Read NLU result from a standalone JSON file.",
    )
    parser.add_argument(
        "--retrieval-file",
        type=str,
        default="",
        metavar="FILE",
        help="Read retrieval result from a standalone JSON file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Load demo data from data/sentiment_demo_nlu.json and data/sentiment_demo_retrieval.json.",
    )
    parser.set_defaults(mock=True)
    return parser


def _load_data(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.demo:
        nlu_path = ROOT / "data" / "sentiment_demo_nlu.json"
        retrieval_path = ROOT / "data" / "sentiment_demo_retrieval.json"
        nlu = json.loads(nlu_path.read_text(encoding="utf-8"))
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        print(f"Loaded demo data from: data/sentiment_demo_*.json")
        return nlu, retrieval

    if args.nlu_file and args.retrieval_file:
        nlu = json.loads(Path(args.nlu_file).read_text(encoding="utf-8"))
        retrieval = json.loads(Path(args.retrieval_file).read_text(encoding="utf-8"))
        print(f"Loaded NLU from: {args.nlu_file}")
        print(f"Loaded retrieval from: {args.retrieval_file}")
        return nlu, retrieval

    if args.json_file:
        raw = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        return raw["nlu_result"], raw["retrieval_result"]

    if args.input:
        input_dir = Path(args.input)
        nlu = json.loads((input_dir / "nlu_result.json").read_text(encoding="utf-8"))
        retrieval = json.loads((input_dir / "retrieval_result.json").read_text(encoding="utf-8"))
        print(f"Loaded data from: {input_dir}")
        return nlu, retrieval

    print("Using built-in demo data.")
    return DEMO_NLU, DEMO_RETRIEVAL


def _patch_infer(classifier: SentimentClassifier) -> None:
    """Replace _infer_single with the mock function."""
    from unittest.mock import patch

    patcher = patch.object(classifier, "_infer_single", side_effect=_mock_infer)
    patcher.start()
    # Keep a reference to prevent garbage collection
    classifier._mock_patcher = patcher  # type: ignore[attr-defined]


def _prepare_run_dir(label: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(label)[:48] or "sentiment"
    run_dir = OUTPUT_DIR / f"{timestamp}-sentiment-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", "-", lowered)
    lowered = re.sub(r"[^0-9a-zA-Z一-鿿_-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-_")


def _score_bar(score: float, width: int = 20) -> str:
    """Render a simple ASCII bar for sentiment score visualization."""
    pos = int(score * width)
    bar = "#" * pos + "-" * (width - pos)
    return f"[{bar}]"


if __name__ == "__main__":
    main()
